# SwipeX AI/NLP Service

FastAPI standalone service for resume text parsing, spaCy skill extraction, sentence-transformers embeddings, ChromaDB vector indexing, ATS scoring, Gemini suggestions, and semantic job recommendations.

## Requirements & Architecture

- **Port**: `8001` (default)
- **Framework**: FastAPI (Python 3.11+)
- **Security**: Internal API Key required via header `X-Internal-Api-Key`
- **CPU PyTorch Index**: `https://download.pytorch.org/whl/cpu`

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key variables:
- `ENVIRONMENT`: `dev` | `staging` | `production`
- `INTERNAL_API_KEY`: Secret API key shared with SwipeX backend
- `EMBEDDING_MODEL`: Default `sentence-transformers/all-MiniLM-L6-v2`
- `GEMINI_API_KEY`: Optional Gemini API key for resume optimization suggestions

## Backend Contract

| Endpoint Path | Method | Backend Adapter Method | Description |
|---|---|---|---|
| `/health` | GET | `health_check` | Service liveness probe |
| `/ready` | GET | `readiness_check` | Service readiness probe (models, vector DB, LLM status) |
| `/api/v1/extract/text` | POST | `extract_text` | Extract raw text from PDF/DOCX resume file |
| `/api/v1/extract/skills` | POST | `extract_skills` | Extract skills, experience, and duration from text |
| `/api/v1/extract/resume` | POST | `extract_resume` | End-to-end file parsing and skill extraction |
| `/api/v1/ats/score` | POST | `score_resume_job` | Compute blended ATS match score & missing items |
| `/api/v1/ats/score-upload` | POST | `score_upload` | Inline raw file extraction + ATS match scoring |
| `/api/v1/ats/score/batch` | POST | `score_batch` | Vectorized batch ATS scoring (max 200 jobs) |
| `/api/v1/suggestions` | POST | `generate_suggestions` | Generate AI resume improvement recommendations |
| `/api/v1/recommendations/rank` | POST | `rank_jobs` | Rank candidate jobs using semantic + skill + swipe signals |
| `/api/v1/recommendations/jobs` | POST | `feed_jobs` | Rank jobs directly from ChromaDB index |
| `/api/v1/index/jobs` | POST | `index_jobs` | Asynchronously embed and index job postings |
| `/api/v1/index/tasks/{task_id}` | GET | `get_task_status` | Poll status of asynchronous indexing job |
| `/api/v1/index/stats` | GET | `get_index_stats` | Inspect ChromaDB collection stats and model dimensions |
| `/api/v1/index/reindex` | POST | `reindex_all` | Drop collection and re-index vector store |
| `/api/v1/insights/skill-gap` | POST | `analyze_skill_gap` | Analyze candidate skill gaps against benchmark jobs |
| `/api/v1/insights/resume-performance` | POST | `compute_resume_performance` | Aggregate ATS performance stats & score distribution |
| `/api/v1/insights/hiring-trends` | POST | `compute_hiring_trends` | Compute top demanded skills, experience & work mode trends |

## Running Locally

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

2. Download models:
```bash
python scripts/download_models.py
```

3. Run the development server:
```bash
uvicorn app.main:app --port 8001 --reload
```

## Running Tests & Quality Checks

```bash
pytest tests/ -v
ruff check app/ tests/
```
