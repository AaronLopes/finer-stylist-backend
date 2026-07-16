# Finer Stylist Backend

Flask API for Finer's iOS styling experience. It builds outfits from quiz answers, chat prompts, or live Today context; searches and serves products from Supabase; persists and renders fits; caches shared style personas; and sends APNs push notifications.

## Current capabilities

- Quiz, chat, and Today-context outfit generation with category-aware slot selection.
- Omega-first catalog queries with automatic Chi fallback. Set `PRIMARY_RPC=chi` to reverse the priority.
- Natural-language product search using OpenAI `text-embedding-3-small` embeddings and pgvector cosine similarity.
- Product detail responses with material and provenance-bearing sustainability data.
- Persisted fits, Gemini-generated outfit images, optional selfie-based likeness, and private signed image URLs.
- Shared post-quiz style personas cached by `gender|style|occasion|setting`.
- APNs device registration and admin-gated broadcasts for sandbox and production devices.

## Architecture

```text
api_server.py
  ├── outfit_builder.py         quiz/chat/Today params, category guidance, catalog RPC fallback
  ├── chat_service.py           stylist replies and follow-up intent resolution
  ├── product_search_service.py OpenAI query embeddings and semantic product search
  ├── fit_image_service.py      fit persistence, Gemini image generation, personas, signed URLs
  └── push_service.py           APNs token auth and delivery
```

Outfit slots are queried through Supabase RPCs. The default order is `ff_build_outfit_v3` over `finer_products_omega`, followed by `ff_build_outfit_chi` when the primary source returns no suitable candidate. Category constraints are applied to avoid slot mismatches and are relaxed progressively when needed.

## API endpoints

Routes are mounted at the server root; there is no `/api` prefix.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Validate required environment variables and test Supabase/OpenAI connections |
| `POST` | `/outfit/build` | Unified outfit builder; accepts `mode: "quiz"` or `mode: "chat"` |
| `POST` | `/outfit/build/quiz` | Build directly from quiz answers |
| `POST` | `/outfit/build/chat` | Build from chat and add a stylist reply, follow-up chips, and CTA hint |
| `POST` | `/outfit/build/today` | Build or return the cached daily fit from profile, weather, location, and calendar context |
| `POST` | `/outfit/swap` | Replace one item while preserving the rest of the outfit |
| `POST` | `/chat/parse` | Preview the structured outfit parameters inferred from a chat prompt |
| `POST` | `/chat/resolve-intent` | Resolve an ambiguous follow-up into a standalone outfit query |
| `POST` | `/persona/resolve` | Get or create a cached post-quiz style persona |
| `POST` | `/fits` | Persist a fit without generating images |
| `GET` | `/fits?user_id=<uuid>` | List a user's fits with batched private signed image URLs |
| `POST` | `/fits/generate-images` | Generate images for a new or existing fit and persist them |
| `GET` | `/products/search?q=<text>&limit=20` | Semantic catalog search; `limit` is capped at 50 |
| `GET` | `/products/<product_id>` | Product detail, materials, and sustainability substantiation |
| `POST` | `/devices/register` | Upsert an APNs device token |
| `POST` | `/admin/push/broadcast` | Send an admin-authenticated push to all or selected devices |

### Quiz outfit

```bash
curl -X POST http://localhost:8000/outfit/build/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "feminine",
    "occasion": "date",
    "weather": ["moderate"],
    "setting": "city",
    "goals": ["serve-looks"],
    "style": "minimalist",
    "budget": "$$"
  }'
```

`gender`, `occasion`, `style`, and `budget` are required for quiz requests.

### Chat outfit

```bash
curl -X POST http://localhost:8000/outfit/build/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "something flowy and romantic for a vineyard date, under $200",
    "user_id": "optional-user-uuid",
    "user_profile": {
      "gender": "feminine"
    }
  }'
```

When `user_profile` is sparse and `user_id` is provided, the unified builder attempts to fill missing preferences from the user's latest stored quiz profile.

### Today outfit

```bash
curl -X POST http://localhost:8000/outfit/build/today \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-uuid",
    "profile": {
      "gender": "masculine",
      "style": "classic",
      "budget": "$$"
    },
    "today_context": {
      "temp_f": 68,
      "condition": "sunny",
      "daypart": "evening",
      "city_label": "Brooklyn, NY",
      "local_date": "2026-07-15",
      "primary_event": {
        "title": "Dinner",
        "category": "social",
        "all_day": false
      }
    }
  }'
```

The cache key is `(user_id, local date, inferred occasion)`. On a cache miss the fit is saved to the user's Fits collection; generated image URLs can then be written back through `/fits/generate-images`.

### Product search and detail

```bash
curl "http://localhost:8000/products/search?q=linen+shirt+for+summer&limit=10"
curl "http://localhost:8000/products/PRODUCT_ID"
```

The detail endpoint may return a `null` sustainability score when the available evidence is insufficient. Consumers should present the accompanying `data_completeness`, `score_version`, and `breakdown` instead of treating the score as an unsupported standalone judgment.

### Push broadcast

```bash
curl -X POST http://localhost:8000/admin/push/broadcast \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_PUSH_TOKEN" \
  -d '{
    "title": "A new edit is ready",
    "body": "Open Finer to see the latest looks.",
    "data": {"route": "discover"}
  }'
```

With neither `user_ids` nor `device_tokens`, a broadcast targets every active registered device. Invalid tokens reported by APNs are deactivated automatically.

## Local development

Python 3.11 is used in production.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
# Fill in the credentials needed for the features you are running.

python api_server.py
```

`.env.example` sets `PORT=8000`. If `PORT` is omitted, the development server defaults to `10000`.

Run the test suite with:

```bash
python -m pip install pytest
python -m pytest tests/
```

Most tests isolate external services with fixtures or monkeypatching. The manual integration path in `tests/test_today_endpoint.py` requires live credentials and data.

## Environment variables

### Core

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Service-role key used by the backend |
| `OPENAI_API_KEY` | Yes | — | Chat parsing/composition, tag matching, and product-search embeddings |
| `GEMINI_API_KEY` | For images/personas | — | Gemini outfit and persona image generation |
| `PORT` | No | `10000` | Development server port; Render supplies this value |
| `FLASK_DEBUG` | No | `false` | Enable Flask debug mode |
| `PRIMARY_RPC` | No | `omega` | Catalog priority: `omega` or `chi` |

### Search, chat, and image generation

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_EMBED_MODEL` | `text-embedding-3-small` | Must match the catalog embedding model |
| `SEARCH_RPC` | `match_products_semantic` | Supabase product-search function |
| `CHAT_MODEL` | `gpt-4o-mini` | Stylist reply and intent resolver model |
| `COMPOSER_TIMEOUT_S` | `3.0` | Stylist reply timeout |
| `RESOLVER_TIMEOUT_S` | `2.5` | Follow-up intent resolution timeout |
| `FIT_IMAGES_BUCKET` | `fit-images-private` | Private generated-fit image bucket |
| `FIT_SIGNED_URL_TTL_SECONDS` | `604800` | Signed fit-image URL lifetime in seconds |
| `PERSONAS_BUCKET` | `personas` | Public shared-persona image bucket |

### APNs

| Variable | Required for push | Default | Purpose |
| --- | --- | --- | --- |
| `APNS_KEY_ID` | Yes | — | Apple APNs key ID |
| `APNS_TEAM_ID` | Yes | — | Apple Developer Team ID |
| `APNS_BUNDLE_ID` | No | `app.finer.fit` | APNs topic/bundle ID |
| `APNS_KEY_PATH` | One key source required | — | Path to the APNs `.p8` key |
| `APNS_KEY_P8` | One key source required | — | Inline `.p8` contents instead of a file |
| `ADMIN_PUSH_TOKEN` | For broadcasts | — | Shared secret expected in `X-Admin-Token` |

For Render, upload the `.p8` key as a Secret File. The push service can use `APNS_KEY_PATH`, inline `APNS_KEY_P8`, or auto-discover a single `.p8` file under `/etc/secrets`.

## Supabase requirements

This repository contains incremental migrations. The base product catalog, fit tables, and outfit RPC definitions are maintained in the related `smart-fashion-backend` repository.

Required data objects include:

- Catalog tables: `finer_products_omega` and `finer_products_chi`.
- Outfit RPCs: `ff_build_outfit_v3` and `ff_build_outfit_chi`.
- Semantic helpers: `finer_tag_embeddings`, `match_tags`, and `match_products_semantic`.
- Fit data: `user_fits`, `user_fit_images`, and `finer_daily_fits`.
- Profiles and notifications: `user_profile`, `personas`, and `device_tokens`.
- Storage: private `fit-images-private` and public `personas` buckets, unless overridden by environment variables.

Apply the migrations in `migrations/` as needed:

| Migration | Purpose |
| --- | --- |
| `002_finer_daily_fits_image_urls.sql` | Cache rendered Today images on the daily-fit row |
| `003_chi_drop_merchant_filter.sql` | Documents the required upstream Chi RPC change |
| `004_product_semantic_search.sql` | Configure 1536-dimensional pgvector search and create `match_products_semantic` |
| `005_device_tokens.sql` | Create the service-role-only APNs device registry |
| `006_personas.sql` | Create the shared persona cache and its read policy |

Before applying migration `004`, populate compatible catalog embeddings:

```bash
python scripts/reembed_omega_openai.py --dry-run
python scripts/reembed_omega_openai.py --limit 50
python scripts/reembed_omega_openai.py
```

The script is resumable and must use the same `SEARCH_EMBED_MODEL` as the API.

The product-detail route also expects these columns on `finer_products_omega`: `product_material`, `product_s_score`, `s_score_grade`, `score_data_completeness`, `score_version`, and `score_breakdown`.

## Production deployment

[`render.yaml`](render.yaml) defines a Render web service using Python 3.11 and Gunicorn:

```bash
gunicorn api_server:app --bind 0.0.0.0:$PORT --timeout 300 \
  --workers 2 --threads 4 --worker-class gthread
```

The threaded workers allow Supabase, OpenAI, Gemini, and signed-URL requests to overlap. Fit listing signs all image paths in one storage request to avoid serial URL-signing latency.

## Recent updates reflected in this README

- **July 9, 2026:** batched fit-image signed URLs and added Gunicorn worker/thread concurrency.
- **July 5, 2026:** added product detail with sustainability score substantiation.
- **July 2, 2026:** added shared style-persona caching and APNs registration/broadcast support.
- **June 28, 2026:** added category-guided outfit construction and safer fit-listing timeout behavior.
- **June 16, 2026:** added optional `selfie_url` likeness references for generated fit images.
- **June 15, 2026:** added semantic product search and made Omega the default catalog source with Chi fallback.
