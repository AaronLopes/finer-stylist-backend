-- 008_outfit_rpc_brand.sql
-- (a) Backfill product_brand for legacy rows from the old brand-prefixed
--     product_id shape ("afends000002" = brand + sitemap-order digits).
-- (b) ff_build_outfit_v3 returns product_brand.
-- Run in the Supabase SQL editor after 007.

-- ── 1. Legacy backfill from product_id ──────────────────────────────────
-- Only touches NULL rows, so 007's domain-derived brands are untouched.
-- New-style ids (gohauscom_5102d..., awin_114434_1) don't match the pattern.
-- Caveat: initcap of the raw prefix — multi-word brands come out as one word
-- ("knownsupply000144" -> "Knownsupply"); fix those with targeted UPDATEs.

-- Preview what it would produce:
-- SELECT initcap(substring(product_id from '^([A-Za-z]+)[-_]?[0-9]{1,9}$')) AS brand, count(*)
-- FROM public.finer_products_omega
-- WHERE product_brand IS NULL AND product_id ~ '^[A-Za-z]+[-_]?[0-9]{1,9}$'
-- GROUP BY 1 ORDER BY 2 DESC;

UPDATE public.finer_products_omega
SET product_brand = initcap(substring(product_id from '^([A-Za-z]+)[-_]?[0-9]{1,9}$'))
WHERE product_brand IS NULL
  AND product_id ~ '^[A-Za-z]+[-_]?[0-9]{1,9}$';

-- ── 1b. Stragglers the id pattern missed ────────────────────────────────
-- Digits/hyphens inside the brand prefix (threads4thought, eco-shades,
-- cmmn-swdn) defeat the regex above; Rhude's real domain is rh-ude.com.
UPDATE public.finer_products_omega
SET product_brand = CASE lower(substring(product_link from '^https?://(?:www\.)?([^/:?#]+)'))
    WHEN 'threads4thought.com' THEN 'Threads 4 Thought'
    WHEN 'rh-ude.com'          THEN 'Rhude'
    WHEN 'eco-shades.com'      THEN 'Eco Shades'
    WHEN 'cmmn-swdn.com'       THEN 'CMMN SWDN'
END
WHERE product_brand IS NULL
  AND lower(substring(product_link from '^https?://(?:www\.)?([^/:?#]+)')) IN
      ('threads4thought.com', 'rh-ude.com', 'eco-shades.com', 'cmmn-swdn.com');

-- ── 2. ff_build_outfit_v3 returns product_brand ─────────────────────────
-- Same definition as before + product_brand after product_title in the
-- return table and both SELECT lists. DROP first: return-type change.
DROP FUNCTION IF EXISTS public.ff_build_outfit_v3;

CREATE OR REPLACE FUNCTION public.ff_build_outfit_v3(p_slot text DEFAULT NULL::text, p_gender text DEFAULT NULL::text, p_min_price numeric DEFAULT NULL::numeric, p_max_price numeric DEFAULT NULL::numeric, p_style_tags text[] DEFAULT NULL::text[], p_occasion_tags text[] DEFAULT NULL::text[], p_season_tags text[] DEFAULT NULL::text[], p_avoid_tags text[] DEFAULT NULL::text[], p_target_formality smallint DEFAULT NULL::smallint, p_hero_slot boolean DEFAULT false, p_existing_colors text[] DEFAULT NULL::text[], p_existing_textures text[] DEFAULT NULL::text[], p_color_strategy text DEFAULT 'balanced'::text, p_boost_affiliates boolean DEFAULT true, p_affiliate_boost_weight double precision DEFAULT 0.15, p_include_categories text[] DEFAULT NULL::text[], p_exclude_categories text[] DEFAULT NULL::text[], p_n integer DEFAULT 5)
 RETURNS TABLE(product_id text, product_title text, product_brand text, product_link text, product_img_link text, product_price text, product_price_amount numeric, product_description text, product_slot text, product_category text, product_gender text, product_color text, product_size text, product_tags text[], product_formality smallint, product_role text, product_texture text, affiliate_link text, has_affiliate boolean, tag_match_score double precision, formality_score double precision, color_harmony_score double precision, texture_diversity_score double precision, role_score double precision, affiliate_score double precision, total_score double precision)
 LANGUAGE plpgsql
 STABLE
AS $function$
BEGIN
  RETURN QUERY
  WITH
    -- Pre-compute lowercase arrays ONCE
    params AS (
      SELECT
        lower(p_slot) AS slot_lower,
        lower(p_gender) AS gender_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_style_tags)), ARRAY[]::TEXT[]) AS style_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_occasion_tags)), ARRAY[]::TEXT[]) AS occasion_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_season_tags)), ARRAY[]::TEXT[]) AS season_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_avoid_tags)), ARRAY[]::TEXT[]) AS avoid_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_existing_colors)), ARRAY[]::TEXT[]) AS existing_colors_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_existing_textures)), ARRAY[]::TEXT[]) AS existing_textures_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_include_categories)), ARRAY[]::TEXT[]) AS include_categories_lower,
        COALESCE(ARRAY(SELECT lower(unnest) FROM unnest(p_exclude_categories)), ARRAY[]::TEXT[]) AS exclude_categories_lower
    ),

    -- Get best affiliate link per product (highest commission)
    best_affiliates AS (
      SELECT DISTINCT ON (s.product_id)
        s.product_id,
        s.affiliate_url
      FROM public.finer_products_synthesis s
      INNER JOIN public.finer_affiliate_merchants m ON s.merchant_id = m.id
      WHERE s.is_active = true AND m.is_active = true
      ORDER BY s.product_id, COALESCE(s.commission_rate, m.default_commission_rate) DESC NULLS LAST
    ),

    -- Filter products first (uses indexes)
    base_products AS (
      SELECT
        f.*,
        ba.affiliate_url AS aff_link,
        (ba.affiliate_url IS NOT NULL) AS has_aff
      FROM public.finer_products_omega f
      CROSS JOIN params p
      LEFT JOIN best_affiliates ba ON f.product_id = ba.product_id
      WHERE
        -- Slot filter
        (p_slot IS NULL OR lower(f.product_slot) = p.slot_lower)
        -- Granular category filters
        AND (
          array_length(p.include_categories_lower, 1) IS NULL
          OR lower(f.product_category) = ANY(p.include_categories_lower)
        )
        AND (
          array_length(p.exclude_categories_lower, 1) IS NULL
          OR lower(coalesce(f.product_category, '')) <> ALL(p.exclude_categories_lower)
        )
        -- Gender filter. NULL / genderless means 'do not filter'; masculine
        -- and feminine are exact product-label matches.
        AND (
          p_gender IS NULL
          OR p.gender_lower = 'genderless'
          OR lower(f.product_gender) = p.gender_lower
        )
        -- Price filter
        AND (
          (p_min_price IS NULL AND p_max_price IS NULL)
          OR f.product_price_amount BETWEEN COALESCE(p_min_price, 0) AND COALESCE(p_max_price, 1000000)
        )
        -- Must NOT have avoid tags
        AND (
          array_length(p.avoid_lower, 1) IS NULL
          OR array_length(p.avoid_lower, 1) = 0
          OR NOT (f.product_tags && p.avoid_lower)
        )
        -- Must have at least one style OR occasion OR season tag (if any provided)
        -- No strict gym filter - we boost gym products in scoring instead
        AND (
          (
            (array_length(p.style_lower, 1) IS NULL OR array_length(p.style_lower, 1) = 0)
            AND (array_length(p.occasion_lower, 1) IS NULL OR array_length(p.occasion_lower, 1) = 0)
            AND (array_length(p.season_lower, 1) IS NULL OR array_length(p.season_lower, 1) = 0)
          )
          OR f.product_tags && p.style_lower
          OR f.product_tags && p.occasion_lower
          OR f.product_tags && p.season_lower
        )
    ),

    -- Score each product
    scored_products AS (
      SELECT
        b.product_id,
        b.product_title,
        b.product_brand,
        b.product_link,
        b.product_img_link,
        b.product_price,
        b.product_price_amount,
        b.product_description,
        b.product_slot,
        b.product_category,
        b.product_gender,
        b.product_color,
        b.product_size,
        b.product_tags,
        b.product_formality,
        b.product_role,
        b.product_texture,
        b.aff_link,
        b.has_aff,

        -- Tag match score: proportion of requested tags that product has
        -- Plus bonus for gym/activewear products when gym occasion selected
        (CASE
          WHEN array_length(p.style_lower || p.occasion_lower || p.season_lower, 1) > 0 THEN
            COALESCE(
              array_length(
                ARRAY(
                  SELECT unnest(b.product_tags)
                  INTERSECT
                  SELECT unnest(p.style_lower || p.occasion_lower || p.season_lower)
                ), 1
              )::FLOAT / array_length(p.style_lower || p.occasion_lower || p.season_lower, 1)::FLOAT,
              0.0
            )
          ELSE 0.5
        END
        -- Boost gym/activewear products when gym occasion is selected (+0.4 bonus)
        + CASE
            WHEN 'gym' = ANY(p.occasion_lower)
                 AND b.product_tags && ARRAY['gym', 'activewear', 'athletic', 'workout']::TEXT[]
            THEN 0.4
            ELSE 0.0
          END
        )::FLOAT AS tag_match_score,

        -- Formality score: closer to target = higher (0.2 penalty per level difference)
        CASE
          WHEN p_target_formality IS NOT NULL AND b.product_formality IS NOT NULL THEN
            GREATEST(0.0, 1.0 - (ABS(b.product_formality - p_target_formality)::FLOAT * 0.2))
          ELSE 0.5
        END::FLOAT AS formality_score,

        -- Color harmony score based on strategy
        CASE
          -- Neutral strategy: prefer neutrals
          WHEN p_color_strategy = 'neutral' THEN
            CASE
              WHEN lower(b.product_color) IN ('black', 'white', 'gray', 'grey', 'beige', 'navy', 'brown', 'cream', 'tan', 'ivory', 'charcoal') THEN 1.0
              ELSE 0.3
            END
          -- Bold strategy: prefer non-neutrals
          WHEN p_color_strategy = 'bold' THEN
            CASE
              WHEN lower(b.product_color) IN ('black', 'white', 'gray', 'grey', 'beige') THEN 0.4
              ELSE 0.9
            END
          -- Monochrome strategy: match existing colors
          WHEN p_color_strategy = 'monochrome' AND array_length(p.existing_colors_lower, 1) > 0 THEN
            CASE
              WHEN lower(b.product_color) = ANY(p.existing_colors_lower) THEN 1.0
              WHEN lower(b.product_color) IN ('black', 'white', 'gray', 'grey') THEN 0.7
              ELSE 0.2
            END
          -- Balanced (default): neutrals good, avoid exact repeats
          ELSE
            CASE
              WHEN lower(b.product_color) IN ('black', 'white', 'gray', 'grey', 'beige', 'navy', 'brown') THEN 0.7
              WHEN lower(b.product_color) = ANY(p.existing_colors_lower) THEN 0.3  -- penalize exact repeat
              ELSE 0.6
            END
        END::FLOAT AS color_harmony_score,

        -- Texture diversity score: penalize if texture already in outfit
        CASE
          WHEN array_length(p.existing_textures_lower, 1) > 0 AND b.product_texture IS NOT NULL THEN
            CASE
              WHEN lower(b.product_texture) = ANY(p.existing_textures_lower) THEN 0.3  -- penalize repeat
              ELSE 0.8  -- reward diversity
            END
          ELSE 0.5  -- neutral if no context
        END::FLOAT AS texture_diversity_score,

        -- Role score: boost statement pieces for hero slots, basics otherwise
        CASE
          WHEN p_hero_slot = true THEN
            CASE
              WHEN b.product_role = 'statement' THEN 0.9
              WHEN b.product_role = 'basic' THEN 0.4
              ELSE 0.6
            END
          ELSE
            CASE
              WHEN b.product_role = 'basic' THEN 0.8
              WHEN b.product_role = 'statement' THEN 0.5
              ELSE 0.6
            END
        END::FLOAT AS role_score,

        -- Affiliate score: slight boost for products with affiliate links
        -- Gap is small (0.7 vs 0.5) so affiliates get a nudge, not domination
        CASE
          WHEN p_boost_affiliates AND b.has_aff THEN 0.7
          WHEN p_boost_affiliates AND NOT b.has_aff THEN 0.5
          ELSE 0.5  -- neutral if not boosting
        END::FLOAT AS affiliate_score

      FROM base_products b
      CROSS JOIN params p
    ),

    -- Calculate total score with weights
    final_scored AS (
      SELECT
        s.*,
        -- Weighted total score with random factor for variety
        -- Random adds 0-0.15 variance so similar products shuffle
        (
          (s.tag_match_score * (0.30 - CASE WHEN p_boost_affiliates THEN p_affiliate_boost_weight/2 ELSE 0 END)) +
          (s.formality_score * 0.25) +
          (s.color_harmony_score * 0.20) +
          (s.texture_diversity_score * 0.10) +
          (s.role_score * (0.15 - CASE WHEN p_boost_affiliates THEN p_affiliate_boost_weight/2 ELSE 0 END)) +
          (s.affiliate_score * CASE WHEN p_boost_affiliates THEN p_affiliate_boost_weight ELSE 0 END) +
          (RANDOM() * 0.15)  -- Add randomness for variety
        )::FLOAT AS total_score
      FROM scored_products s
    )

  SELECT
    f.product_id,
    f.product_title,
    f.product_brand,
    f.product_link,
    f.product_img_link,
    f.product_price,
    f.product_price_amount,
    f.product_description,
    f.product_slot,
    f.product_category,
    f.product_gender,
    f.product_color,
    f.product_size,
    f.product_tags,
    f.product_formality,
    f.product_role,
    f.product_texture,
    f.aff_link AS affiliate_link,
    f.has_aff AS has_affiliate,
    f.tag_match_score,
    f.formality_score,
    f.color_harmony_score,
    f.texture_diversity_score,
    f.role_score,
    f.affiliate_score,
    f.total_score
  FROM final_scored f
  ORDER BY f.total_score DESC, RANDOM()
  LIMIT GREATEST(1, COALESCE(p_n, 5));

END;
$function$;
