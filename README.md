# Finer Stylist Backend

Flask API server for AI-powered outfit building. Supports both quiz-based and chat-based outfit generation.

## Endpoints

| Method | Endpoint                 | Description                                   |
| ------ | ------------------------ | --------------------------------------------- |
| GET    | `/api/health`            | Health check                                  |
| POST   | `/api/outfit/build`      | Build outfit (unified - quiz or chat mode)    |
| POST   | `/api/outfit/build/quiz` | Build outfit from quiz answers                |
| POST   | `/api/outfit/build/chat` | Build outfit from natural language            |
| POST   | `/api/outfit/swap`       | Swap single item in outfit                    |
| POST   | `/api/chat/parse`        | Parse chat query (preview what we understood) |

## Quick Start

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run development server
python api_server.py
```

### Production (Render)

The app is configured for Render deployment. Set these environment variables in Render:

- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_KEY` - Your Supabase service role key
- `OPENAI_API_KEY` - Your OpenAI API key

## API Usage

### Build Outfit (Quiz Mode)

```bash
curl -X POST https://your-app.onrender.com/api/outfit/build/quiz \
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

### Build Outfit (Chat Mode)

```bash
curl -X POST https://your-app.onrender.com/api/outfit/build/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "something flowy and romantic for a vineyard date, under $200",
    "user_profile": {
      "gender": "feminine"
    }
  }'
```

### Swap Item

```bash
curl -X POST https://your-app.onrender.com/api/outfit/swap \
  -H "Content-Type: application/json" \
  -d '{
    "current_outfit": { ... },
    "slot_to_swap": "top",
    "exclude_product_ids": ["current_top_id"]
  }'
```

### Parse Chat Query

```bash
curl -X POST https://your-app.onrender.com/api/chat/parse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "casual friday outfit for the office"
  }'
```

## Response Format

### Outfit Response

```json
{
  "success": true,
  "items": {
    "top": {
      "product_id": "...",
      "product_title": "...",
      "product_link": "...",
      "product_img_link": "...",
      "product_price": "$99",
      "product_price_amount": 99.00,
      "product_color": "black",
      "product_tags": ["minimalist", "classic"],
      "total_score": 0.85
    },
    "bottom": { ... },
    "footwear": { ... }
  },
  "total_price": 245.00,
  "params_used": {
    "gender": "feminine",
    "occasion": "date",
    "style_tags": ["minimalist", "solid", "neutral"],
    "color_strategy": "bold"
  }
}
```

## Environment Variables

| Variable         | Description                        |
| ---------------- | ---------------------------------- |
| `SUPABASE_URL`   | Supabase project URL               |
| `SUPABASE_KEY`   | Supabase service role key          |
| `OPENAI_API_KEY` | OpenAI API key                     |
| `PORT`           | Server port (default: 8000)        |
| `FLASK_DEBUG`    | Enable debug mode (default: false) |

## Database Requirements

This API requires the following Supabase setup:

1. **Table**: `finer_products_omega` with columns:
   - `product_formality` (SMALLINT)
   - `product_role` (TEXT)
   - `product_texture` (TEXT)

2. **Table**: `finer_tag_embeddings` for semantic tag matching

3. **Function**: `ff_build_outfit_v2` for outfit scoring

4. **Function**: `match_tags` for semantic tag matching

See the `smart-fashion-backend` repo for SQL migrations.
