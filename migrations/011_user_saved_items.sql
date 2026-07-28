-- 011_user_saved_items.sql
-- Saved products shown in the iOS Closet.
--
-- The product fields are copied onto each saved row so the item can still be
-- shown if it is later removed from the product catalog.

create table if not exists public.user_saved_items (
  id          uuid not null default gen_random_uuid(),
  user_id     uuid not null,
  product_id  text null,
  name        text not null,
  brand       text null,
  price       numeric null,
  image_url   text null,
  link        text null,
  category    text null,
  source      text null,
  created_at  timestamptz not null default now(),
  constraint user_saved_items_pkey primary key (id),
  constraint user_saved_items_user_id_product_id_key
    unique (user_id, product_id),
  constraint user_saved_items_user_id_fkey
    foreign key (user_id) references auth.users (id) on delete cascade
);

alter table public.user_saved_items enable row level security;

revoke all on table public.user_saved_items from anon;
grant select, insert, update, delete on table public.user_saved_items
  to authenticated;

-- Replace the original generic policy names if the iOS-owned SQL was applied.
drop policy if exists "own rows select" on public.user_saved_items;
drop policy if exists "own rows insert" on public.user_saved_items;
drop policy if exists "own rows update" on public.user_saved_items;
drop policy if exists "own rows delete" on public.user_saved_items;

drop policy if exists "saved items select own rows" on public.user_saved_items;
create policy "saved items select own rows"
  on public.user_saved_items
  for select
  to authenticated
  using ((select auth.uid()) = user_id);

drop policy if exists "saved items insert own rows" on public.user_saved_items;
create policy "saved items insert own rows"
  on public.user_saved_items
  for insert
  to authenticated
  with check ((select auth.uid()) = user_id);

drop policy if exists "saved items update own rows" on public.user_saved_items;
create policy "saved items update own rows"
  on public.user_saved_items
  for update
  to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

drop policy if exists "saved items delete own rows" on public.user_saved_items;
create policy "saved items delete own rows"
  on public.user_saved_items
  for delete
  to authenticated
  using ((select auth.uid()) = user_id);
