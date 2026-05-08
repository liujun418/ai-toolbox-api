# AI ToolBox API

FastAPI backend for AI ToolBox Online (ai.toolboxonline.club).

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

- `GET /` — Health check
- `GET /api/tasks/health` — API health
- `GET /api/tasks/{task_id}` — Get task status
- `POST /api/upload/{tool_type}` — Upload file and start processing

## Environment Variables

See `.env.example` for required variables:

- `DATABASE_URL` — PostgreSQL connection string
- `REPLICATE_API_TOKEN` — Replicate API key
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — S3/R2 credentials
- `AWS_S3_ENDPOINT` — Leave empty for AWS S3, set for Cloudflare R2
- `STRIPE_SECRET_KEY` — Stripe API key
- `STRIPE_WEBHOOK_SECRET` — Stripe webhook signing secret
- `FRONTEND_URL` — Frontend origin for CORS
