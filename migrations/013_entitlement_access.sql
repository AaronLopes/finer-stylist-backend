-- Preserve the existing billing schema and rows. Only trusted servers may
-- write entitlements; signed-in clients may read their own account's grant.
begin;
alter table public.entitlements enable row level security;
drop policy if exists "Service role can manage entitlements" on public.entitlements;
drop policy if exists "Users can view own entitlements" on public.entitlements;
revoke all on public.entitlements from public, anon, authenticated;
grant select on public.entitlements to authenticated;
grant all on public.entitlements to service_role;
create policy "Users can view own entitlements" on public.entitlements
 for select to authenticated using ((select auth.uid()) = user_id);
commit;
