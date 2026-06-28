# Vigilo — ML-Powered Ad Campaign Intelligence

> *Vigilo* (Latin) — "I watch over"

Vigilo analyses Google Ads campaigns using machine learning and tells you exactly where your budget is being wasted — and what to do about it.

## Live Demo
[![Live Demo](https://img.shields.io/badge/Live_Demo-Vigilo-D4A853?style=for-the-badge)](https://YOUR-VERCEL-URL.vercel.app)

[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github)](https://github.com/AkshitSonkusale/Vigilo)

---

## ML Pipeline

```
CSV Upload → Feature Engineering → K-Means Clustering → Isolation Forest → Health Score → Groq API
```

| Stage | What it does |
|---|---|
| Feature Engineering | Normalizes CTR, CPC, CVR, ROAS, spend with StandardScaler |
| K-Means (k=4) | Groups campaigns: Underinvested Winner / Bleeding Budget / Steady Performer / Keyword Waste |
| Isolation Forest | Flags statistical outliers — splits into wasteful anomalies vs positive standouts |
| Health Score | Deterministic 0–100 formula from cluster + anomaly + metric signals |
| Groq API | Cluster-aware natural language recommendations per campaign |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, React (CDN), Lekton + Space Grotesk |
| Backend | FastAPI, Python |
| ML | scikit-learn (K-Means, Isolation Forest), pandas, numpy |
| Database | PostgreSQL via Supabase |
| AI | Groq API (Llama 3.1-8b-instant) |
| Deployment | Vercel (frontend), Render (backend) |

---

## Local Setup

### Backend

```bash
cd backend
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run the server
uvicorn main:app --reload --port 8000
```

### Frontend

Open `frontend/index.html` directly in your browser.  
No build step needed — React loads from CDN.

### Environment Variables

```
DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres
GROQ_API_KEY=gsk_...
FRONTEND_URL=https://your-app.vercel.app
```

### Database Setup

Run `backend/db/schema.sql` once in your Supabase SQL editor to create all tables.

---

## SQL Analytics Layer

Three analytical queries run before the ML pipeline:

1. **Account aggregations** — total spend, conversions, averages
2. **Window functions** — `RANK() OVER`, `PERCENT_RANK() OVER`, CPC ratio vs account average
3. **CTE pre-flagging** — zero conversion, high CPC, low CTR flags

The pipeline falls back to in-memory processing if `DATABASE_URL` is not set — so the demo always works.

---

## Project Structure

```
vigilo/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── pipeline.py          # ML orchestrator
│   ├── models.py            # Pydantic response models
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── clustering.py    # K-Means
│   │   ├── anomaly.py       # Isolation Forest
│   │   ├── health_score.py  # Deterministic scoring
│   │   └── recommendations.py  # Groq API
│   ├── db/
│   │   ├── schema.sql       # Table definitions
│   │   └── queries.py       # SQL analytics layer
│   ├── utils/
│   │   └── csv_parser.py    # Robust CSV parsing
│   └── routes/
│       └── upload.py        # API endpoints
├── frontend/
│   └── index.html           # Complete frontend (no build step)
└── notebooks/
    └── analysis.ipynb       # EDA + model derivation
```

---

## Key Design Decisions

- **Health score is deterministic** — the LLM generates recommendation text only, never scores
- **SQL before ML** — aggregations and window functions run in PostgreSQL; Python receives enriched data
- **Isolation Forest direction fix** — statistical outliers are split into wasteful anomalies vs standout performers; pure statistical outliers flagged the best campaigns as problems
- **Graceful fallback** — no DATABASE_URL or API key? The pipeline still runs in-memory with rule-based recommendations

---

*Built by Akshit Sonkusale*
