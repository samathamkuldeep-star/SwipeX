# SwipeX — AI-Powered Job Discovery & Career Assistance Platform

This repository contains the complete codebase for **SwipeX**, including the **AI/NLP Career Intelligence Service (FastAPI)**.

## 📦 GitHub Repository
```bash
git clone https://github.com/samathamkuldeep-star/SwipeX.git
cd SwipeX/swipex-ai-service
```

---

## 🚀 For Frontend & UX Developers (Intern 1)

If you are developing the frontend components (Notification bell, Tracking board, Analytics dashboards, and Charts), the key contracts and schemas are located inside `swipex-ai-service/`:

### 1. Alert Triggers (Day 2–3)
- **High-Match Score Alert**: Triggered when `match_score >= 70.0` (from `/api/v1/ats/score`).
- **Startup Hiring Alerts**: Triggered when `job.company_type` is `"startup"` or `"newly_founded"`.
- **Low-Competition Alerts**: Triggered when `job.competition_level == "low"` or `applicant_count < 10`.
- **Reference Schemas**:
  - `app/schemas/ats.py` (`ATSResult`)
  - `app/schemas/job.py` (`JobInput`)

### 2. AI Analytics Endpoints & Schemas (Day 3–4)
| Feature | Endpoint Path | DTO Schemas | Schema File |
|---|---|---|---|
| **Skill-Gap Analysis** | `POST /api/v1/insights/skill-gap` | `SkillGapRequest`, `SkillGapResponse` | `app/schemas/insights.py` |
| **Resume Performance** | `POST /api/v1/insights/resume-performance` | `ResumePerformanceRequest`, `ResumePerformanceResponse` | `app/schemas/insights.py` |
| **Hiring Trends Charts** | `POST /api/v1/insights/hiring-trends` | `HiringTrendsRequest`, `HiringTrendsResponse` | `app/schemas/insights.py` |
| **Recommendation Feed** | `POST /api/v1/recommendations/jobs` | `FeedJobsRequest`, `FeedJobsResponse` | `app/schemas/recommendations.py` |

---

## 🛠️ Running the AI Service Locally

```bash
cd swipex-ai-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/download_models.py
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive API documentation (Swagger) is available at:
👉 **http://localhost:8001/docs**
