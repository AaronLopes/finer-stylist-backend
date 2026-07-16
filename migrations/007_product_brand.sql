-- 007_product_brand.sql
-- Brand column for finer_products_omega, derived from product_link's domain.
-- Re-run section 2 after scraping new products (no trigger). Add a brand =
-- add a WHEN arm. Run in the Supabase SQL editor.

-- ── 1. Column ───────────────────────────────────────────────────────────
ALTER TABLE public.finer_products_omega ADD COLUMN IF NOT EXISTS product_brand text;

-- ── 2. Backfill (idempotent; only touches NULL rows) ────────────────────
UPDATE public.finer_products_omega
SET product_brand = CASE lower(substring(product_link from '^https?://(?:www\.)?([^/:?#]+)'))
    WHEN 'gohaus.com'          THEN 'Haus'
    WHEN 'aloyoga.com'         THEN 'Alo'          -- TODO verify
    WHEN 'stussy.com'          THEN 'Stussy'       -- TODO verify
    WHEN 'elwoodclothing.com'  THEN 'Elwood'       -- TODO verify
    WHEN 'madhappy.com'        THEN 'Madhappy'     -- TODO verify
    WHEN 'vuoriclothing.com'   THEN 'Vuori'        -- TODO verify
    WHEN 'fearofgod.com'       THEN 'Fear of God'  -- TODO verify
    WHEN 'rh-ude.com'          THEN 'Rhude'
    WHEN 'agjeans.com'         THEN 'AG Jeans'     -- TODO verify
    WHEN 'cottonon.com'        THEN 'Cotton On'    -- TODO verify
    WHEN 'talentless.co'       THEN 'Talentless'   -- TODO verify
    WHEN 'afends.com'          THEN 'Afends'       -- TODO verify (legacy)
    ELSE NULL
END
WHERE product_brand IS NULL
  AND product_link IS NOT NULL;

-- Unmapped domains (don't map affiliate redirect domains — leave NULL):
-- SELECT lower(substring(product_link from '^https?://(?:www\.)?([^/:?#]+)')) AS domain, count(*)
-- FROM public.finer_products_omega
-- WHERE product_brand IS NULL AND product_link IS NOT NULL
-- GROUP BY 1 ORDER BY 2 DESC;

-- ── 3. Search RPC returns product_brand (drop first: return-type change) ─
DROP FUNCTION IF EXISTS match_products_semantic(extensions.vector, int);

CREATE FUNCTION match_products_semantic(
  query_embedding extensions.vector(1536),
  match_count int DEFAULT 20
)
RETURNS TABLE(
  product_id text,
  product_title text,
  product_link text,
  product_img_link text,
  product_price_amount numeric,
  product_slot text,
  product_description text,
  product_brand text,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    f.product_id,
    f.product_title,
    f.product_link,
    f.product_img_link,
    f.product_price_amount,
    f.product_slot,
    f.product_description,
    f.product_brand,
    1 - (f.product_embedding <=> query_embedding) AS similarity
  FROM public.finer_products_omega f
  WHERE f.product_embedding IS NOT NULL
    AND f.product_title IS NOT NULL
    AND f.product_img_link IS NOT NULL
  ORDER BY f.product_embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ── 4. Outfit RPCs ──────────────────────────────────────────────────────
-- ff_build_outfit_v3 (and any siblings) must also return product_brand.
-- Dump the current definitions, then recreate each (DROP first) with
-- product_brand added to RETURNS TABLE and the SELECT list:
-- SELECT proname, pg_get_functiondef(oid) FROM pg_proc WHERE proname LIKE 'ff_build_outfit%';
