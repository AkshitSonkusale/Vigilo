# Vigilo – ML-Powered Ad Campaign Optimization Platform

Vigilo analyzes Google Ads campaign data through a 5-stage ML pipeline to identify underperforming campaigns, detect anomalies, and generate AI-powered optimization recommendations.

---

## Pipeline Overview

1. **Data Ingestion** — CSV parsing and validation of raw Google Ads campaign data
2. **Feature Engineering** — StandardScaler normalization across 5 campaign features
3. **K-Means Clustering** — Segments campaigns into 4 clusters (elbow method + silhouette score evaluation), with human-readable cluster labels derived from centroids
4. **Isolation Forest Anomaly Detection** — Identifies wasteful anomalies vs standout performers via a domain-anchored directional business rule
5. **Health Score + Recommendations** — Deterministic 0–100 health score per campaign, with Claude API generating prioritized natural language recommendations ranked by severity

---

## SQL Analytics Layer

Built in PostgreSQL using window functions, CTEs, and aggregations to preprocess raw campaign data before ML input — deliberately separating the analytics layer from the ML pipeline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Pipeline | Python, Scikit-learn, Pandas |
| Backend | FastAPI (deployed on Render) |
| Frontend | React + TypeScript (deployed on Vercel) |
| Database | PostgreSQL |
| AI Layer | Claude API (claude-sonnet-4-6) |
| Notebook | Jupyter (EDA, elbow method, evaluation) |

---

## Running Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

A bundled sample dataset is included for instant demo without needing real Google Ads data.

---

## Key Design Decisions

- **K-Means over DBSCAN** — campaign data is roughly spherical in feature space; K-Means gives more interpretable, stable clusters for a business audience
- **Isolation Forest for anomaly detection** — unsupervised, handles high-dimensional feature space well, no labeled anomaly data required
- **Directional business rule fix** — raw Isolation Forest treats all anomalies equally; added a rule separating high-spend-low-performance (wasteful) from low-spend-high-performance (standout) anomalies
- **Deterministic health score** — chose a formula-based score over model output to ensure explainability and consistency across runs
