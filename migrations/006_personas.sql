-- 006_personas.sql
-- Style-persona cache for the post-quiz "bring your results to life" moment.
--
-- At the end of onboarding the iOS app calls POST /persona/resolve with the
-- user's quiz answers. The backend renders a single full-body model image
-- wearing an outfit built for that style (reusing fit_image_service's Gemini
-- pipeline) and caches it here, keyed by a COARSE combination of the answers:
--
--     persona_key = "<gender>|<style>|<occasion>|<setting>"   (all lowercased)
--
-- Because the key is coarse (~500 possible combinations, not the full quiz
-- product space), personas are heavily REUSED: the first user to land on a
-- combination pays the ~15-30s render cost, everyone after gets the cached
-- image instantly. Weather / budget / goals still shape the OUTFIT BUILD but
-- deliberately do not fork the cached image.
--
-- Unlike user_fits (private, per-user, signed URLs), personas are shared and
-- non-sensitive, so their images live in a PUBLIC storage bucket with
-- permanent URLs (no signed-URL expiry to manage for a long-lived cache).
--
-- The backend (service-role key) is the only writer. The table is also
-- anon-readable so the client *could* pre-check the cache directly, mirroring
-- the read-only access pattern used for finer_products_omega. No auth.users FK:
-- personas are not owned by anyone, and this endpoint runs during onboarding
-- BEFORE the user has an account (guest flow).
--
-- Run in the Supabase SQL editor as a single execution.

create table if not exists public.personas (
    persona_key   text primary key,
    gender        text,
    style         text,
    occasion      text,
    setting       text,
    title         text,
    subtitle      text,
    image_path    text not null,
    image_url     text not null,
    items         jsonb       not null default '[]'::jsonb,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

alter table public.personas enable row level security;

-- Anon/authenticated clients may READ the cache (mirrors finer_products_omega).
-- Writes are service-role only (bypasses RLS); no insert/update/delete policy.
drop policy if exists "personas are publicly readable" on public.personas;
create policy "personas are publicly readable"
    on public.personas
    for select
    to anon, authenticated
    using (true);

-- ---------------------------------------------------------------------------
-- Storage bucket (create via Dashboard → Storage, or the snippet below):
--
--   insert into storage.buckets (id, name, public)
--   values ('personas', 'personas', true)
--   on conflict (id) do update set public = true;
--
-- Must be PUBLIC so image_url resolves to a permanent, CDN-cacheable URL of
-- the form: {SUPABASE_URL}/storage/v1/object/public/personas/<path>.
-- ---------------------------------------------------------------------------
