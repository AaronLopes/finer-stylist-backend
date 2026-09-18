from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from andrea_entitlements import SupabaseEntitlements, EntitlementLookupError

NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)

class Query:
    def __init__(self, rows): self.rows=rows; self.filters=[]
    def table(self, name): assert name=='entitlements'; return self
    def select(self, fields): assert fields=='is_pro,expires_at'; return self
    def eq(self, name, value): self.filters.append((name,value)); return self
    def execute(self): return SimpleNamespace(data=self.rows)

@pytest.mark.parametrize('rows,expected', [
    ([],False),
    ([{'is_pro':False,'expires_at':None}],False),
    ([{'is_pro':True,'expires_at':None}],True),
    ([{'is_pro':True,'expires_at':'2026-09-19T00:00:00Z'}],True),
    ([{'is_pro':True,'expires_at':'2026-09-18T00:00:00Z'}],False),
    ([{'is_pro':True,'expires_at':'2026-09-17T00:00:00Z'}],False),
    ([{'is_pro':'true','expires_at':None}],False),
    ([{'is_pro':True,'expires_at':'2026-09-17T00:00:00Z'}, {'is_pro':True,'expires_at':None}],True),
])
def test_existing_entitlements_are_account_scoped_and_expiry_aware(rows,expected):
    q=Query(rows)
    assert SupabaseEntitlements(lambda:q, now=lambda:NOW).is_pro('verified-user-id') is expected
    assert q.filters==[('user_id','verified-user-id')]

@pytest.mark.parametrize('rows', [None, [{'is_pro':True,'expires_at':'invalid'}], [{'is_pro':True,'expires_at':'2026-09-19T00:00:00'}]])
def test_malformed_or_failed_lookup_never_becomes_a_paywall(rows):
    with pytest.raises(EntitlementLookupError): SupabaseEntitlements(lambda:Query(rows)).is_pro('uid')


def test_native_fallback_is_optional_and_cannot_override_active_web_entitlement():
    fallback=SimpleNamespace(available=lambda:True,is_pro=lambda uid:True)
    assert SupabaseEntitlements(lambda:Query([]),fallback=fallback).is_pro('uid')
    fallback.is_pro=lambda uid:False
    assert SupabaseEntitlements(lambda:Query([{'is_pro':True,'expires_at':None}]),fallback=fallback).is_pro('uid')
