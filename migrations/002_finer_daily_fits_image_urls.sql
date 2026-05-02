-- Cache the Gemini-composed outfit images on the daily-fit row so re-opening
-- Today within the same (user, date, occasion) returns the styled image
-- inline instead of re-running Gemini.
--
-- Apply via psql or the Supabase SQL editor:
--   psql "$SUPABASE_DB_URL" -f migrations/002_finer_daily_fits_image_urls.sql

ALTER TABLE public.finer_daily_fits
    ADD COLUMN IF NOT EXISTS image_urls jsonb;
