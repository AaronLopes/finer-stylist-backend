-- 004_product_semantic_search.sql
-- Semantic product search for GET /products/search (api_server.py).
--
-- PREREQUISITE: run scripts/reembed_omega_openai.py first. It overwrites
-- finer_products_omega.product_embedding with 1536-dim OpenAI
-- text-embedding-3-small vectors on every row (replacing the sparse 768-dim
-- fashionSigLIP set, which only covered ~9% of the catalog and ranked
-- poorly for text queries). The API server embeds queries with the same
-- OpenAI model, so no torch/open_clip needed at serving time.
--
-- Run in the Supabase SQL editor as a single execution so the SETs apply.
--
-- Named match_products_semantic (not match_products) because a legacy
-- match_products function from the old search_server.py era still exists
-- with an incompatible (1536-dim, different return shape) signature;
-- CREATE OR REPLACE cannot change a return type, and dropping it would
-- break old scrape_engine tooling.

-- Index builds need headroom; Supabase allows raising these per-session.
SET statement_timeout = '30min';
SET maintenance_work_mem = '512MB';

-- ── 1. Pin the embedding column to 1536 dims ───────────────────────────
-- The column was created untyped (extensions.vector); pgvector can only
-- index a typed column. Clear any stragglers the re-embed script missed
-- (e.g. rows added mid-run) so the cast can't fail — the script is
-- idempotent and can be rerun to fill them in.
UPDATE public.finer_products_omega
SET product_embedding = NULL
WHERE product_embedding IS NOT NULL
  AND vector_dims(product_embedding) <> 1536;

ALTER TABLE public.finer_products_omega
  ALTER COLUMN product_embedding TYPE extensions.vector(1536)
  USING product_embedding::extensions.vector(1536);

-- ── 2. ANN index (HNSW, cosine) ────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_products_omega_embedding_hnsw
  ON public.finer_products_omega
  USING hnsw (product_embedding extensions.vector_cosine_ops);

-- ── 3. Search function ─────────────────────────────────────────────────
-- Top-N products by cosine similarity to the query embedding. The ORDER BY
-- uses the <=> operator directly against the query constant so the HNSW
-- index is used.
CREATE OR REPLACE FUNCTION match_products_semantic(
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
    1 - (f.product_embedding <=> query_embedding) AS similarity
  FROM public.finer_products_omega f
  WHERE f.product_embedding IS NOT NULL
    AND f.product_title IS NOT NULL
    AND f.product_img_link IS NOT NULL
  ORDER BY f.product_embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Sanity check (a zero vector should still return match_count rows):
-- SELECT product_id, product_title, similarity
-- FROM match_products_semantic(array_fill(0, ARRAY[1536])::extensions.vector(1536), 5);
