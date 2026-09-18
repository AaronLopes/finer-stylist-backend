from uuid import uuid4
import pytest
from andrea_rollout import DEFAULT_RATING_PREVIEW_USER_IDS, ratings_rollout_allows


@pytest.fixture(autouse=True)
def private_preview(monkeypatch):
    monkeypatch.setenv('ANDREA_RATINGS_ENABLED','false')
    monkeypatch.delenv('ANDREA_RATINGS_PREVIEW_USER_IDS',raising=False)


def test_default_preview_is_exactly_one_owner_account():
    assert len(DEFAULT_RATING_PREVIEW_USER_IDS)==1
    owner=next(iter(DEFAULT_RATING_PREVIEW_USER_IDS))
    assert ratings_rollout_allows(owner.upper())
    assert not ratings_rollout_allows(str(uuid4()))


@pytest.mark.parametrize('configured',['', 'not-a-uuid', '*,', 'aaronlopes@me.com'])
def test_empty_or_invalid_override_disables_preview(monkeypatch,configured):
    monkeypatch.setenv('ANDREA_RATINGS_PREVIEW_USER_IDS',configured)
    assert not ratings_rollout_allows(next(iter(DEFAULT_RATING_PREVIEW_USER_IDS)))


def test_configured_ids_replace_default_and_normalize_case(monkeypatch):
    other=str(uuid4())
    monkeypatch.setenv('ANDREA_RATINGS_PREVIEW_USER_IDS',' '+other.upper()+' ,')
    assert ratings_rollout_allows(other)
    assert not ratings_rollout_allows(next(iter(DEFAULT_RATING_PREVIEW_USER_IDS)))


def test_public_rollout_still_requires_a_valid_authenticated_id(monkeypatch):
    monkeypatch.setenv('ANDREA_RATINGS_ENABLED','true')
    assert ratings_rollout_allows(str(uuid4()))
    for invalid in [None,'','not-a-uuid']:
        assert not ratings_rollout_allows(invalid)
