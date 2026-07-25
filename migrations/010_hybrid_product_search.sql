-- 010_hybrid_product_search.sql
-- Hybrid retrieval for GET /products/search.
--
-- Embeddings remain the fuzzy-recall signal, while populated Omega fields
-- make high-confidence intent deterministic:
--   category -> hard filter (with parent-category expansion from Python)
--   color    -> hard filter
--   brand    -> strong lexical rank; hard filter for explicit/bare brands
--
-- The legacy match_products_semantic function remains in place so the API can
-- fall back safely while this migration reaches Supabase's schema cache.

CREATE OR REPLACE FUNCTION public.match_products_hybrid(
  query_embedding extensions.vector(1536) DEFAULT NULL,
  match_count int DEFAULT 20,
  p_query text DEFAULT '',
  p_query_terms text[] DEFAULT NULL,
  p_categories text[] DEFAULT NULL,
  p_colors text[] DEFAULT NULL,
  p_brand_query text DEFAULT NULL,
  p_brand_terms text[] DEFAULT NULL,
  p_require_brand boolean DEFAULT false,
  p_diversify_categories boolean DEFAULT false,
  p_gender text DEFAULT NULL
)
RETURNS TABLE(
  product_id text,
  product_title text,
  product_link text,
  product_img_link text,
  product_price_amount numeric,
  product_slot text,
  product_category text,
  product_color text,
  product_description text,
  product_brand text,
  similarity float,
  search_score float
)
LANGUAGE sql
STABLE
AS $$
  WITH params AS (
    SELECT
      lower(btrim(coalesce(p_query, ''))) AS query_lower,
      regexp_replace(
        lower(coalesce(p_brand_query, '')),
        '[^a-z0-9]+',
        '',
        'g'
      ) AS brand_query_compact,
      ARRAY(
        SELECT lower(btrim(term))
        FROM unnest(coalesce(p_query_terms, ARRAY[]::text[])) AS term
        WHERE length(btrim(term)) >= 2
      ) AS query_terms,
      ARRAY(
        SELECT lower(btrim(category))
        FROM unnest(coalesce(p_categories, ARRAY[]::text[])) AS category
        WHERE btrim(category) <> ''
      ) AS categories,
      ARRAY(
        SELECT lower(btrim(color))
        FROM unnest(coalesce(p_colors, ARRAY[]::text[])) AS color
        WHERE btrim(color) <> ''
      ) AS colors,
      ARRAY(
        SELECT lower(btrim(term))
        FROM unnest(coalesce(p_brand_terms, ARRAY[]::text[])) AS term
        WHERE length(btrim(term)) >= 2
      ) AS brand_terms
  ),
  searchable AS (
    -- Keep the full 1536-dim vector out of this repeatedly referenced CTE.
    -- Materializing vectors for the whole catalog on every query would erase
    -- the latency benefit of the pgvector candidate pass below.
    SELECT
      f.product_id,
      f.product_title,
      f.product_brand,
      f.product_slot,
      f.product_category,
      f.product_color,
      f.product_tags
    FROM public.finer_products_omega AS f
    WHERE f.product_title IS NOT NULL
      AND f.product_img_link IS NOT NULL
      AND (
        p_gender IS NULL
        OR lower(p_gender) = 'genderless'
        OR lower(coalesce(f.product_gender, 'genderless'))
          IN ('genderless', lower(p_gender))
      )
  ),
  semantic_ids AS (
    SELECT f.product_id
    FROM public.finer_products_omega AS f
    WHERE query_embedding IS NOT NULL
      AND f.product_embedding IS NOT NULL
      AND f.product_title IS NOT NULL
      AND f.product_img_link IS NOT NULL
      AND (
        p_gender IS NULL
        OR lower(p_gender) = 'genderless'
        OR lower(coalesce(f.product_gender, 'genderless'))
          IN ('genderless', lower(p_gender))
      )
    ORDER BY f.product_embedding <=> query_embedding
    LIMIT greatest(100, least(500, coalesce(match_count, 20) * 5))
  ),
  annotated AS (
    SELECT
      f.*,
      p.categories,
      p.colors,
      CASE
        WHEN lower(coalesce(f.product_category, '')) = 'pants'
          AND words.title_words
            ~ '(^| )(trouser|trousers|slack|slacks|chino|chinos)( |$)'
          THEN 'trousers'
        WHEN lower(coalesce(f.product_category, '')) = 'pants'
          AND words.title_words
            ~ '(^| )(jogger|joggers|sweatpant|sweatpants|trackpant|trackpants)( |$)'
          THEN 'joggers'
        WHEN lower(coalesce(f.product_category, '')) = 'pants'
          AND words.title_words ~ '(^| )(cargo|cargos)( |$)'
          THEN 'cargo_pants'
        ELSE lower(coalesce(f.product_category, 'uncategorized'))
      END AS diversity_group,
      (
        cardinality(p.categories) > 0
        AND lower(coalesce(f.product_category, '')) = ANY(p.categories)
      ) AS category_match,
      (
        cardinality(p.colors) > 0
        AND lower(btrim(coalesce(f.product_color, ''), ' .')) = ANY(p.colors)
      ) AS color_match,
      (
        p.brand_query_compact <> ''
        AND length(words.brand_compact) >= 2
        AND (
          words.brand_compact = p.brand_query_compact
          OR words.brand_compact LIKE (p.brand_query_compact || '%')
          OR p.brand_query_compact LIKE (words.brand_compact || '%')
        )
      ) AS brand_exact_match,
      EXISTS (
        SELECT 1
        FROM unnest(p.brand_terms) AS brand_term
        WHERE
          (' ' || words.brand_words || ' ') LIKE ('% ' || brand_term || ' %')
          OR words.brand_compact LIKE (brand_term || '%')
      ) AS brand_term_match,
      (
        p.query_lower <> ''
        AND position(p.query_lower IN lower(coalesce(f.product_title, ''))) > 0
      ) AS title_phrase_match,
      EXISTS (
        SELECT 1
        FROM unnest(p.query_terms) AS query_term
        WHERE (' ' || words.search_words || ' ')
          LIKE ('% ' || query_term || ' %')
      ) AS lexical_term_match
    FROM searchable AS f
    CROSS JOIN params AS p
    CROSS JOIN LATERAL (
      SELECT
        regexp_replace(
          lower(coalesce(f.product_title, '')),
          '[^a-z0-9]+',
          ' ',
          'g'
        ) AS title_words,
        regexp_replace(
          lower(coalesce(f.product_brand, '')),
          '[^a-z0-9]+',
          ' ',
          'g'
        ) AS brand_words,
        regexp_replace(
          lower(coalesce(f.product_brand, '')),
          '[^a-z0-9]+',
          '',
          'g'
        ) AS brand_compact,
        regexp_replace(
          lower(concat_ws(
            ' ',
            f.product_title,
            f.product_brand,
            f.product_category,
            f.product_slot,
            f.product_color,
            array_to_string(f.product_tags, ' ')
          )),
          '[^a-z0-9]+',
          ' ',
          'g'
        ) AS search_words
    ) AS words
  ),
  structured_scored AS (
    SELECT
      a.*,
      (
        (a.category_match::int * 3)
        + (a.color_match::int * 3)
        + (a.brand_exact_match::int * 5)
        + (a.brand_term_match::int * 3)
        + (a.title_phrase_match::int * 2)
        + a.lexical_term_match::int
      ) AS structured_score
    FROM annotated AS a
    WHERE a.category_match
      OR a.color_match
      OR a.brand_exact_match
      OR a.brand_term_match
      OR a.title_phrase_match
      OR a.lexical_term_match
  ),
  structured_ranked AS (
    SELECT
      s.*,
      row_number() OVER (
        PARTITION BY s.diversity_group
        ORDER BY s.structured_score DESC, s.product_id
      ) AS diversity_candidate_rank
    FROM structured_scored AS s
  ),
  structured_ids AS (
    SELECT s.product_id
    FROM structured_ranked AS s
    ORDER BY
      CASE
        WHEN p_diversify_categories AND s.diversity_candidate_rank <= 20
          THEN s.diversity_candidate_rank
        ELSE 21
      END,
      s.structured_score DESC,
      s.product_id
    LIMIT greatest(100, least(500, coalesce(match_count, 20) * 5))
  ),
  candidate_ids AS (
    SELECT product_id FROM semantic_ids
    UNION
    SELECT product_id FROM structured_ids
  ),
  scored AS (
    SELECT
      f.product_id,
      f.product_title,
      f.product_link,
      f.product_img_link,
      f.product_price_amount,
      f.product_slot,
      f.product_category,
      f.product_color,
      f.product_description,
      f.product_brand,
      a.categories,
      a.colors,
      a.diversity_group,
      a.category_match,
      a.color_match,
      a.brand_exact_match,
      a.brand_term_match,
      a.title_phrase_match,
      a.lexical_term_match,
      vector_score.similarity,
      (
        (coalesce(vector_score.similarity, 0.0) * 0.62)
        + (a.category_match::int * 0.22)
        + (a.color_match::int * 0.24)
        + CASE
            WHEN a.brand_exact_match THEN 0.48
            WHEN a.brand_term_match THEN 0.32
            ELSE 0.0
          END
        + (a.title_phrase_match::int * 0.15)
        + (a.lexical_term_match::int * 0.06)
      )::float AS search_score
    FROM annotated AS a
    INNER JOIN candidate_ids AS c USING (product_id)
    INNER JOIN public.finer_products_omega AS f USING (product_id)
    CROSS JOIN LATERAL (
      SELECT CASE
        WHEN query_embedding IS NULL OR f.product_embedding IS NULL THEN NULL
        ELSE (1 - (f.product_embedding <=> query_embedding))::float
      END AS similarity
    ) AS vector_score
    WHERE
      (cardinality(a.categories) = 0 OR a.category_match)
      AND (cardinality(a.colors) = 0 OR a.color_match)
      AND (
        NOT p_require_brand
        OR a.brand_exact_match
        OR a.brand_term_match
      )
  ),
  ranked AS (
    SELECT
      s.*,
      row_number() OVER (
        PARTITION BY s.diversity_group
        ORDER BY s.search_score DESC, s.product_id
      ) AS category_rank
    FROM scored AS s
  )
  SELECT
    r.product_id,
    r.product_title,
    r.product_link,
    r.product_img_link,
    r.product_price_amount,
    r.product_slot,
    r.product_category,
    r.product_color,
    r.product_description,
    r.product_brand,
    r.similarity,
    r.search_score
  FROM ranked AS r
  ORDER BY
    CASE
      WHEN p_diversify_categories AND r.category_rank <= 2
        THEN r.category_rank
      ELSE 3
    END,
    r.search_score DESC,
    r.similarity DESC NULLS LAST,
    r.product_id
  LIMIT greatest(1, least(coalesce(match_count, 20), 50));
$$;

COMMENT ON FUNCTION public.match_products_hybrid(
  extensions.vector,
  int,
  text,
  text[],
  text[],
  text[],
  text,
  text[],
  boolean,
  boolean,
  text
) IS
  'Hybrid product search: vector recall plus exact category/color and lexical brand ranking.';

-- Optional smoke checks after applying this migration:
-- SELECT product_title, product_brand, product_category, search_score
-- FROM match_products_hybrid(
--   NULL, 20, 'Kowtow', ARRAY['kowtow'], NULL, NULL,
--   'kowtow', ARRAY['kowtow'], true, false, 'feminine'
-- );
--
-- SELECT product_title, product_brand, product_category, search_score
-- FROM match_products_hybrid(
--   NULL, 20, 'black jeans', ARRAY['black','jeans'],
--   ARRAY['jeans'], ARRAY['black'], NULL, NULL, false, false, 'feminine'
-- );
