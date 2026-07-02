-- 005_device_tokens.sql
-- APNs device-token registry for push notifications (see push_service.py).
--
-- The iOS app registers its APNs device token via POST /devices/register on
-- every launch/permission-grant. The backend (service-role key) is the only
-- writer/reader, so RLS is enabled with NO policies: anon/authenticated
-- clients are denied, the service role bypasses RLS. The app never touches
-- this table directly through the Supabase SDK — it goes through the Flask
-- endpoint, matching the "backend is system of record" split.
--
-- user_id is NULLABLE on purpose: guest devices (no auth session) can still
-- receive marketing/announcement broadcasts. It references auth.users so an
-- account deletion nulls the column (ON DELETE SET NULL) rather than orphaning
-- or dropping the token — the device keeps receiving broadcasts until it
-- unregisters or APNs reports it Unregistered.
--
-- environment distinguishes APNs hosts: Xcode/dev builds mint 'sandbox'
-- tokens (api.sandbox.push.apple.com); TestFlight/App Store builds mint
-- 'production' tokens (api.push.apple.com). The same physical device yields
-- DIFFERENT tokens per environment, so both can legitimately coexist.
--
-- Run in the Supabase SQL editor as a single execution.

create table if not exists public.device_tokens (
    device_token        text primary key,
    user_id             uuid references auth.users(id) on delete set null,
    platform            text        not null default 'ios',
    environment         text        not null default 'production'
                        check (environment in ('production', 'sandbox')),
    app_version         text,
    is_active           boolean     not null default true,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now(),
    last_registered_at  timestamptz not null default now()
);

-- Broadcast queries scan active tokens and route by environment.
create index if not exists device_tokens_active_env_idx
    on public.device_tokens (environment)
    where is_active;

-- Targeted sends ("notify this user across their devices") filter by user_id.
create index if not exists device_tokens_user_idx
    on public.device_tokens (user_id)
    where user_id is not null;

alter table public.device_tokens enable row level security;
-- No policies: only the service-role backend reads/writes this table.
