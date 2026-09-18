-- Andrea keeps no conversation archive or source photos. Lifetime counters and
-- idempotency metadata survive resets; recovery payloads are short-lived.
create table if not exists public.andrea_user_state (
 user_id uuid primary key references auth.users(id) on delete cascade,
 welcome_seen_at timestamptz,
 free_ratings_used integer not null default 0 check (free_ratings_used between 0 and 3)
);
create table if not exists public.andrea_rating_requests (
 user_id uuid not null references auth.users(id) on delete cascade,
 request_id uuid not null,
 fingerprint text not null,
 status text not null check (status in ('processing','completed','retake','failed')),
 access_basis text not null check (access_basis in ('free','pro')),
 attempt integer not null default 1,
 lease_token uuid not null,
 lease_until timestamptz not null,
 result jsonb,
 result_expires_at timestamptz,
 created_at timestamptz not null default now(),
 primary key(user_id, request_id)
);
create table if not exists public.andrea_request_budgets (
 bucket text not null,
 subject text not null,
 window_start bigint not null,
 used integer not null default 0,
 primary key(bucket, subject, window_start)
);
alter table public.andrea_user_state enable row level security;
alter table public.andrea_rating_requests enable row level security;
alter table public.andrea_request_budgets enable row level security;
revoke all on public.andrea_user_state, public.andrea_rating_requests, public.andrea_request_budgets from anon, authenticated;
grant all on public.andrea_user_state, public.andrea_rating_requests, public.andrea_request_budgets to service_role;

create or replace function public.andrea_state(p_user_id uuid, p_mark_read boolean default false)
returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare s public.andrea_user_state;
begin
 insert into andrea_user_state(user_id) values(p_user_id) on conflict do nothing;
 if p_mark_read then
  update andrea_user_state set welcome_seen_at=coalesce(welcome_seen_at, now()) where user_id=p_user_id;
 end if;
 -- Expire recovery content during normal traffic; never retain a chat archive.
 update andrea_rating_requests set result=null where result_expires_at < now() and result is not null;
 select * into s from andrea_user_state where user_id=p_user_id;
 return jsonb_build_object('welcome_seen',s.welcome_seen_at is not null,'free_ratings_remaining',3-s.free_ratings_used);
end $$;

create or replace function public.andrea_reserve_rating(p_user_id uuid, p_request_id uuid, p_fingerprint text, p_is_pro boolean, p_lease_token uuid)
returns jsonb language plpgsql security definer set search_path = public, pg_temp as $$
declare s public.andrea_user_state; r public.andrea_rating_requests; active_count integer;
begin
 insert into andrea_user_state(user_id) values(p_user_id) on conflict do nothing;
 select * into s from andrea_user_state where user_id=p_user_id for update;
 select * into r from andrea_rating_requests where user_id=p_user_id and request_id=p_request_id;
 if found then
  if r.fingerprint <> p_fingerprint then return jsonb_build_object('status','conflict'); end if;
  if r.status in ('completed','retake') then
   return jsonb_build_object('status',case when r.result_expires_at > now() then r.status else 'expired' end,
    'result',case when r.result_expires_at > now() then r.result else null end);
  end if;
  if r.status='processing' and r.lease_until > now() then return jsonb_build_object('status','processing'); end if;
  if r.attempt >= 3 then return jsonb_build_object('status','exhausted'); end if;
 end if;
 update andrea_rating_requests set status='failed' where user_id=p_user_id and status='processing' and lease_until <= now();
 select count(*) into active_count from andrea_rating_requests where user_id=p_user_id and status='processing';
 -- One analysis in flight per account, including Pro, bounds concurrent spend.
 if active_count > 0 then return jsonb_build_object('status','busy'); end if;
 if not p_is_pro and s.free_ratings_used >= 3 then return jsonb_build_object('status','pro_required'); end if;
 insert into andrea_rating_requests(user_id,request_id,fingerprint,status,access_basis,lease_token,lease_until)
 values(p_user_id,p_request_id,p_fingerprint,'processing',case when p_is_pro then 'pro' else 'free' end,p_lease_token,now()+interval '3 minutes')
 on conflict(user_id,request_id) do update set status='processing', access_basis=excluded.access_basis,
  lease_token=excluded.lease_token, lease_until=excluded.lease_until, attempt=andrea_rating_requests.attempt+1;
 return jsonb_build_object('status','reserved');
end $$;

create or replace function public.andrea_finish_rating(p_user_id uuid, p_request_id uuid, p_lease_token uuid, p_status text, p_result jsonb)
returns boolean language plpgsql security definer set search_path = public, pg_temp as $$
declare r public.andrea_rating_requests;
begin
 if p_status not in ('completed','retake','failed') then raise exception 'Invalid rating state'; end if;
 perform 1 from andrea_user_state where user_id=p_user_id for update;
 select * into r from andrea_rating_requests where user_id=p_user_id and request_id=p_request_id for update;
 if not found or r.status <> 'processing' or r.lease_token <> p_lease_token then return false; end if;
 if p_status='completed' and r.access_basis='free' then
  update andrea_user_state set free_ratings_used=free_ratings_used+1 where user_id=p_user_id;
 end if;
 update andrea_rating_requests set status=p_status,result=p_result,result_expires_at=now()+interval '1 hour'
 where user_id=p_user_id and request_id=p_request_id;
 return true;
end $$;

create or replace function public.andrea_get_rating(p_user_id uuid,p_request_id uuid)
returns jsonb language sql security definer set search_path = public, pg_temp as $$
 select jsonb_build_object('status',case when status in ('completed','retake') and result_expires_at <= now() then 'expired' else status end,
  'result',case when result_expires_at > now() then result else null end)
 from andrea_rating_requests where user_id=p_user_id and request_id=p_request_id;
$$;

create or replace function public.andrea_clear_recovery(p_user_id uuid)
returns void language sql security definer set search_path = public, pg_temp as $$
 update andrea_rating_requests set result=null,result_expires_at=now() where user_id=p_user_id;
$$;

create or replace function public.andrea_claim_budget(p_bucket text,p_subject text,p_limit integer,p_window_seconds integer)
returns boolean language plpgsql security definer set search_path = public, pg_temp as $$
declare w bigint; n integer;
begin
 if p_limit < 1 or p_window_seconds < 1 then return false; end if;
 w := floor(extract(epoch from now()) / p_window_seconds)::bigint * p_window_seconds;
 delete from andrea_request_budgets where window_start < extract(epoch from now()) - 172800;
 insert into andrea_request_budgets(bucket,subject,window_start,used) values(p_bucket,p_subject,w,1)
 on conflict(bucket,subject,window_start) do update set used=andrea_request_budgets.used+1
 where andrea_request_budgets.used < p_limit returning used into n;
 return n is not null;
end $$;

revoke all on function public.andrea_state(uuid,boolean), public.andrea_reserve_rating(uuid,uuid,text,boolean,uuid), public.andrea_finish_rating(uuid,uuid,uuid,text,jsonb), public.andrea_get_rating(uuid,uuid), public.andrea_clear_recovery(uuid), public.andrea_claim_budget(text,text,integer,integer) from public,anon,authenticated;
grant execute on function public.andrea_state(uuid,boolean), public.andrea_reserve_rating(uuid,uuid,text,boolean,uuid), public.andrea_finish_rating(uuid,uuid,uuid,text,jsonb), public.andrea_get_rating(uuid,uuid), public.andrea_clear_recovery(uuid), public.andrea_claim_budget(text,text,integer,integer) to service_role;
