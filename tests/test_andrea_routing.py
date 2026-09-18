"""Provider-boundary coverage for conversational outfit-card routing."""
import json
from types import SimpleNamespace
import pytest
from andrea_service import AndreaService

PROFILE = {'occasion':'work', 'style':'classic', 'gender':'masculine'}
HISTORY = [
    {'role':'user','content':"Build a look around white sneakers under $150. Ask me where I'm headed."},
    {'role':'assistant','content':'Where are you headed for work?'},
]


def service_returning(value, calls):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=None,choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(value)))])
    return AndreaService(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))


@pytest.mark.parametrize('rating_context', [None, {'score':8,'observations':['A navy jacket']}])
def test_latest_turn_is_separate_from_profile_and_prior_conversation(rating_context):
    calls=[]
    answer={'kind':'outfit','outfit_query':'Build a date outfit around white sneakers under $150.','reply_text':''}
    service=service_returning(answer,calls)
    assert service.chat('Date',HISTORY,PROFILE,None,rating_context)==answer
    assert len(calls)==1
    turns=calls[0]['messages']
    assert turns[0]['role']=='system'
    assert turns[2:-1]==HISTORY
    assert turns[-1]=={'role':'user','content':'Date'}
    context=json.loads(turns[1]['content'].removeprefix('Background context only: '))
    assert context['profile_defaults']==PROFILE
    assert context['last_rating']==rating_context


def test_refinement_retains_current_outfit_and_advice_stays_supported():
    calls=[]
    service=service_returning({'kind':'advice','reply_text':'The restrained trousers let the jacket lead.'},calls)
    outfit='Request: Date outfit. Items: navy jacket, gray trousers.'
    result=service.chat('Why this jacket?',[],PROFILE,outfit,None)
    assert result['kind']=='advice'
    context=json.loads(calls[0]['messages'][1]['content'].removeprefix('Background context only: '))
    assert context['current_outfit']==outfit


@pytest.mark.parametrize('value', [
    {'kind':'other'}, {'kind':'outfit','outfit_query':''},
    {'kind':'advice','reply_text':''}, {'kind':'advice','reply_text':None},
])
def test_incomplete_routing_response_is_rejected(value):
    with pytest.raises(ValueError):
        service_returning(value,[]).chat('Date',HISTORY,PROFILE,None,None)
