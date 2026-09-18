"""Additive Andrea routes. Auth, quotas and costs are enforced on the server."""
import hashlib
import io
import json
import logging
import os
from uuid import UUID, uuid4
from urllib.parse import quote

import requests
from flask import Blueprint, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from andrea_service import AndreaService, short_text
from andrea_entitlements import SupabaseEntitlements, EntitlementLookupError

logger = logging.getLogger(__name__)
MAX_PHOTO_BYTES = 6 * 1024 * 1024
PROFILE_FIELDS = ('gender','occasion','season','setting','goals','style','budget')


class AndreaError(Exception):
    def __init__(self, code, message, status=503):
        self.code, self.message, self.status = code, message, status


class SuperwallEntitlements:
    """Live lookup: no stale local entitlement cache or second purchase SDK.

    Source: api.superwall.com/docs, usersv2.getUserActiveEntitlements.
    The existing iOS identity is the Supabase UUID string (uppercase on Swift).
    """
    def available(self):
        return all(os.getenv(k) for k in ('SUPERWALL_SERVER_API_KEY','SUPERWALL_APPLICATION_ID','SUPERWALL_PRO_ENTITLEMENT'))

    def is_pro(self, user_id):
        if not self.available():
            raise AndreaError('subscription_unavailable','Ratings are being set up. Please try again shortly.')
        try:
            response = requests.get('https://api.superwall.com/v2/users/'+quote(user_id.upper(),safe='')+'/active-entitlements',
                params={'application_id':os.environ['SUPERWALL_APPLICATION_ID']},
                headers={'Authorization':'Bearer '+os.environ['SUPERWALL_SERVER_API_KEY']},timeout=8)
            response.raise_for_status()
            data=response.json()
            if not isinstance(data.get('entitlements'),list) or any(not isinstance(x,str) for x in data['entitlements']):
                raise ValueError('Invalid entitlement response')
            return os.environ['SUPERWALL_PRO_ENTITLEMENT'] in data['entitlements']
        except (requests.RequestException,ValueError,TypeError) as exc:
            raise AndreaError('subscription_unavailable','I could not check your subscription. Please try again; no rating was used.') from exc


class AndreaStore:
    def __init__(self, client): self.client=client
    def rpc(self,name,**params): return self.client.rpc('andrea_'+name,params).execute().data
    def state(self,uid,read=False): return self.rpc('state',p_user_id=uid,p_mark_read=read)
    def budget(self,bucket,subject,limit,seconds):
        if not self.rpc('claim_budget',p_bucket=bucket,p_subject=subject,p_limit=limit,p_window_seconds=seconds):
            raise AndreaError('rate_limited','Andrea is taking a short breather. Please try again later.',429)
    def get(self,uid,rid): return self.rpc('get_rating',p_user_id=uid,p_request_id=rid)
    def reserve(self,uid,rid,fingerprint,pro,lease):
        return self.rpc('reserve_rating',p_user_id=uid,p_request_id=rid,p_fingerprint=fingerprint,p_is_pro=pro,p_lease_token=lease)
    def finish(self,uid,rid,lease,status,result):
        return self.rpc('finish_rating',p_user_id=uid,p_request_id=rid,p_lease_token=lease,p_status=status,p_result=result)


def normalize_photo(data):
    try:
        with Image.open(io.BytesIO(data)) as photo:
            if photo.width*photo.height > 25_000_000 or photo.width < 32 or photo.height < 32:
                raise ValueError('Invalid dimensions')
            photo=ImageOps.exif_transpose(photo).convert('RGB')
            photo.thumbnail((1280,1280))
            output=io.BytesIO()
            photo.save(output,format='JPEG',quality=85)
            return output.getvalue()
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as exc:
        raise AndreaError('invalid_photo','Please choose a clear JPEG, PNG, or WebP outfit photo.',400) from exc


def clean_profile(value):
    if not isinstance(value,dict): return {}
    profile = {key:short_text(value.get(key),100) for key in PROFILE_FIELDS if isinstance(value.get(key),str)}
    for key in ('goals','weather'):
        if isinstance(value.get(key),list):
            profile[key]=[short_text(x,80) for x in value[key][:8] if isinstance(x,str)]
    return profile


def clean_context(value,limit):
    if not isinstance(value,(dict,list,str)): return None
    if len(json.dumps(value)) > limit:
        raise AndreaError('invalid_request','The conversation is too long. Please start a new chat.',400)
    return value


def register_andrea_routes(app, builder_provider, supabase_provider, *, service_provider=None, entitlement_provider=None, store_provider=None):
    bp=Blueprint('andrea',__name__,url_prefix='/andrea')
    service=None
    entitlement=entitlement_provider or SupabaseEntitlements(supabase_provider, fallback=SuperwallEntitlements())
    def ai():
        nonlocal service
        if service_provider: return service_provider()
        if service is None: service=AndreaService()
        return service
    def store(): return store_provider() if store_provider else AndreaStore(supabase_provider())
    def identity(required=True):
        authorization=request.headers.get('Authorization','')
        if not authorization and not required: return None
        if not authorization.startswith('Bearer ') or len(authorization)>10000:
            raise AndreaError('sign_in_required','Sign in to use your three free outfit ratings.',401)
        try:
            user=supabase_provider().auth.get_user(authorization[7:]).user
            return str(UUID(str(user.id)))
        except Exception as exc:
            raise AndreaError('sign_in_required','Please sign in again to continue.',401) from exc
    def request_id(value):
        try: return str(UUID(value))
        except (ValueError,TypeError,AttributeError) as exc:
            raise AndreaError('invalid_request','Missing or invalid request ID.',400) from exc
    def ratings_enabled():
        return os.getenv('ANDREA_RATINGS_ENABLED','false').lower()=='true' and entitlement.available()
    def access_state(uid,read=False):
        state=store().state(uid,read)
        pro=None
        if ratings_enabled():
            try: pro=entitlement.is_pro(uid)
            except (AndreaError, EntitlementLookupError): pass
        return {**state,'pro_status':'active' if pro else ('inactive' if pro is False else 'unavailable'),
                'ratings_enabled':ratings_enabled() and pro is not None}
    def budget(kind,uid):
        s=store()
        subject=uid or hashlib.sha256((request.remote_addr or 'unknown').encode()).hexdigest()
        s.budget(kind+'_minute',subject,12 if uid else 30,60)
        s.budget(kind+'_day','global',int(os.getenv('ANDREA_'+kind.upper()+'_DAILY_LIMIT','500' if kind=='rating' else '5000')),86400)
    def recovery_response(uid,row):
        if row and row.get('status') in ('completed','retake') and row.get('result'):
            return jsonify(success=True,rating=row['result'],state=access_state(uid))
        status=row.get('status') if row else 'not_found'
        errors={
            'processing':('rating_processing','Your outfit is still being assessed. Try again in a moment.',409),
            'busy':('rating_processing','Another outfit is being assessed. Please wait for it to finish.',409),
            'conflict':('request_conflict','This request ID belongs to a different photo.',409),
            'exhausted':('retry_exhausted','Please start a new rating request.',409),
            'expired':('result_expired','This temporary result has expired. Start a new rating if needed.',410),
            'pro_required':('pro_required','You have used your three free ratings. Continue with Pro.',402),
            'failed':('rating_failed','That rating could not finish. Retry your photo; no free rating was used.',503),
            'not_found':('not_found','Rating not found.',404)}
        raise AndreaError(*errors.get(status,errors['failed']))

    @bp.before_request
    def bounds():
        request.max_content_length=MAX_PHOTO_BYTES+65536 if request.path.endswith('/ratings') else 32000
        if request.content_length and request.content_length>request.max_content_length:
            raise RequestEntityTooLarge()

    @bp.errorhandler(AndreaError)
    def expected(exc): return jsonify(success=False,error=exc.message,code=exc.code),exc.status
    @bp.errorhandler(EntitlementLookupError)
    def subscription_failed(exc):
        return expected(AndreaError('subscription_unavailable','I could not check your subscription. Please try again; no rating was used.'))

    @bp.errorhandler(Exception)
    def failed(exc):
        if isinstance(exc,HTTPException):
            return jsonify(success=False,error='The request could not be read.',code='invalid_request'),exc.code
        logger.error('Andrea request failed type=%s',type(exc).__name__)
        return jsonify(success=False,error='Andrea could not finish that request. Please try again.',code='service_unavailable'),503

    @bp.get('/state')
    def state(): return jsonify(success=True,state=access_state(identity()))

    @bp.post('/welcome/read')
    def welcome(): return jsonify(success=True,state=access_state(identity(),True))

    @bp.post('/reset')
    def reset():
        uid=identity()
        store().rpc('clear_recovery',p_user_id=uid)
        return jsonify(success=True)

    @bp.post('/chat')
    def chat():
        uid=identity(False)
        data=request.get_json(silent=True)
        if not isinstance(data,dict): raise AndreaError('invalid_request','Please send a message.',400)
        message=short_text(data.get('message'),2000)
        if not message: raise AndreaError('invalid_request','Please send a message.',400)
        history=data.get('history',[])
        if not isinstance(history,list): raise AndreaError('invalid_request','Invalid conversation.',400)
        history=[{'role':m['role'],'content':short_text(m.get('content'),700)} for m in history[-10:]
                 if isinstance(m,dict) and m.get('role') in ('user','assistant')]
        profile=clean_profile(data.get('profile'))
        outfit_context=clean_context(data.get('outfit_context'),5000)
        rating_context=clean_context(data.get('rating_context'),4000)
        budget('chat',uid)
        answer=ai().chat(message,history,profile,outfit_context,rating_context)
        if answer['kind']=='advice':
            return jsonify(success=True,kind='advice',reply_text=short_text(answer['reply_text'],1000),actions=[])
        query=short_text(answer['outfit_query'],600)
        outfit=builder_provider().build_from_chat(query,user_profile=profile)
        outfit={'success':True,**outfit}
        try: reply=ai().describe_outfit(query,outfit,profile)
        except Exception: reply='Here is the look I put together for you.'
        if not any(outfit.get('items',{}).values()):
            return jsonify(success=True,kind='advice',reply_text="I couldn't find the right pieces. Try another occasion or style direction.",actions=[])
        return jsonify(success=True,kind='outfit',reply_text=reply,outfit=outfit,
            actions=[{'id':'message','label':'Make it more casual','message':'Make this outfit more casual'},
                     {'id':'message','label':'Try another look','message':'Build another outfit for the same occasion'}])

    @bp.get('/ratings/<rid>')
    def get_rating(rid):
        uid=identity()
        return recovery_response(uid,store().get(uid,request_id(rid)))

    @bp.post('/ratings')
    def rating():
        uid=identity()
        if not ratings_enabled(): raise AndreaError('ratings_unavailable','Photo ratings are not available yet. You can still chat with Andrea.')
        rid=request_id(request.form.get('request_id'))
        photo=request.files.get('image')
        if photo is None or photo.mimetype not in ('image/jpeg','image/png','image/webp'):
            raise AndreaError('invalid_photo','Choose a JPEG, PNG, or WebP outfit photo.',400)
        data=photo.stream.read(MAX_PHOTO_BYTES+1)
        if len(data)>MAX_PHOTO_BYTES: raise AndreaError('photo_too_large','Please choose a smaller photo.',413)
        image=normalize_photo(data)
        occasion=short_text(request.form.get('occasion'),500)
        try: profile=clean_profile(json.loads(request.form.get('profile','{}')))
        except ValueError as exc: raise AndreaError('invalid_request','Invalid style profile.',400) from exc
        fingerprint=hashlib.sha256(image+json.dumps([occasion,profile],sort_keys=True).encode()).hexdigest()
        s=store()
        # Replays still go through reserve to verify the fingerprint. A completed
        # request can be recovered even after a subscription check becomes unavailable.
        previous=s.get(uid,rid)
        pro=False if previous and previous['status'] in ('completed','retake') else entitlement.is_pro(uid)
        lease=str(uuid4())
        reservation=s.reserve(uid,rid,fingerprint,pro,lease)
        if reservation['status']!='reserved': return recovery_response(uid,reservation)
        try:
            budget('rating',uid)
            result=ai().rate(image,occasion,profile)
            status='completed' if result['rateable'] else 'retake'
            if not s.finish(uid,rid,lease,status,result):
                raise AndreaError('rating_processing','This request changed while processing. Please retry.',409)
        except Exception:
            # Compare-and-set makes this a no-op if the result was already committed.
            try: s.finish(uid,rid,lease,'failed',None)
            except Exception: pass
            raise
        # A state lookup failure must not discard the completed result. Retrying
        # this exact request will recover it without another model call or credit.
        return jsonify(success=True,rating=result,state=access_state(uid))

    app.register_blueprint(bp)
