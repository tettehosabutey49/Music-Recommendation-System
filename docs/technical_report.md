# Music Recommendation System: ML System Design Report

**Creator:** Emmanuel Osabutey | tettehosabutey@outlook.com | [github.com/tettehosabutey49](https://github.com/tettehosabutey49)
**Live Demo:** [Music Recommender on Streamlit](https://jkdmyefxdcigf3suf2axz3.streamlit.app/)
**GitHub:** [github.com/tettehosabutey49/Music-Recommendation-System](https://github.com/tettehosabutey49/Music-Recommendation-System)

---

## ⚠️ Demo vs Production

| | Live Demo (Streamlit Cloud) | Production (Clone & Run Locally) |
|---|---|---|
| **Branch** | `deployment-clean` | `v1.2` / `v1.3` |
| **Data** | Synthetic (10,000 mock songs) | Real music catalog (10,000 actual songs) |
| **Models** | Trained on synthetic data | Trained on authentic listening patterns |
| **Why?** | Streamlit Cloud has a 100MB upload limit; database + models = ~450MB | Full experience when cloned |

The ML algorithms and system architecture are identical — only the data source differs.
The production version includes real songs from The Weeknd, Queen, Eminem, Taylor Swift, Drake, and more across 10 genres, with real audio features (energy, tempo, valence, danceability).

---

## System Overview

Production-grade ensemble recommendation system serving 10,000 songs to 500 users with **<100ms latency**. Combines collaborative filtering (user behaviour patterns), content-based filtering (audio features), and a popularity baseline to handle cold-start and deliver personalised recommendations.

---

## How The Models Work

### 1. Collaborative Filtering (NMF) — 60% weight

Predicts based on what similar users liked.

**Logic:** "Users who liked songs A, B, C also liked song D, so you'll probably like D."

**How it works:** Uses Non-negative Matrix Factorization (NMF) to decompose the user-song interaction matrix into latent user preferences and song characteristics. The model learns entirely from listening patterns — it never looks at song features like genre or artist name.

> **Note:** The class in code is named `ALSRecommender` for API compatibility, but the implementation uses scikit-learn's NMF — a pure-Python alternative that requires no C++ compiler, making setup frictionless across all platforms.

**Trade-offs:**
- **Pro:** Best personalisation once a user has listening history
- **Con:** Fails for new users with no history (cold-start problem)
- **Performance:** Trains in ~5 minutes, inference <5ms

### 2. Content-Based Filtering — 30% weight

Predicts based on song audio features.

**Logic:** "You like high-energy rock songs — here are other high-energy rock songs."

**How it works:** Computes cosine similarity across four audio features — energy (0–1, intensity), tempo (BPM), valence (0–1, happiness/positivity), danceability (0–1). Features are min-max normalised at load time so no single dimension dominates.

**Trade-offs:**
- **Pro:** Solves cold-start — works for new users and new songs with no play history
- **Con:** Can create filter bubbles (always recommends similar-sounding content)
- **Performance:** Similarity matrix is pre-computed at training time, inference <2ms

### 3. Ensemble — Adaptive Weighting

The ensemble adapts its weights based on how much data exists for each user:

| User Stage | Interactions | Weights |
|---|---|---|
| New user | 0 | 70% content-based + 30% popularity |
| Warming | 1–5 | 50% collaborative + 50% content-based |
| Active | 5+ | 60% collaborative + 30% content-based + 10% popularity |

This guarantees **100% user coverage from day 1**, gradually shifting to personalised collaborative filtering as listening history grows.

---

## Key System Design Decisions

### Why NMF instead of Deep Learning?

**Trade-off: Speed vs Accuracy**

| Metric | NMF | Neural Collaborative Filtering |
|---|---|---|
| Training time | ~5 min | 30+ min |
| Accuracy gap at 10K scale | baseline | +2–3% |
| Iteration speed | fast | 6x slower |
| Setup complexity | pip install | GPU + CUDA |

NMF is the right choice for MVP scale. Migration to two-tower neural networks is the documented path at 1M+ songs.

### Cold-Start Handling

**Problem:** New users have no history, making collaborative filtering useless.

**Solution:** Multi-tier fallback.
1. New users get content-based recommendations seeded from popular songs in their preferred genres
2. As they listen, weights shift toward collaborative filtering
3. Result: 100% recommendation coverage from day 1

### Inference Optimisation — <100ms Target

Four techniques that compound:

1. **Sparse matrices** — 98% sparsity reduces memory from ~400MB to ~8MB
2. **Pre-computed embeddings** — user/item NMF factors are calculated during training, not at inference time
3. **Vectorised NumPy operations** — no Python loops in the hot path
4. **Model-level caching** — models load once at startup (2–3s cold start), then serve all requests in <5ms

### Caching Strategy

| Cached | Not Cached |
|---|---|
| NMF factors, song features, metadata | Individual recommendation results |

Individual recommendations are cheap to compute (<10ms) and caching them would reduce personalisation quality (same user would see the same results even as their listening history changes). Cache what is expensive to recompute, not what is cheap.

### Scalability Path

| Scale | Training | Inference | Database | Est. Cost/Month |
|---|---|---|---|---|
| **10K songs (current)** | 5 min | <100ms | SQLite | $0 |
| 100K songs | 30 min | <150ms | PostgreSQL + connection pooling | ~$50 |
| 1M songs | 4+ hours | <200ms | Distributed + Redis | $500+ |

---

## Automated Retraining Pipeline (Airflow) — v1.3

An Apache Airflow DAG automates the full retraining cycle on a **weekly schedule**.

### Pipeline

```
[Every Sunday 00:00 UTC]

  ingest_data
      │  Fetches this week's top tracks from Last.fm (global chart + by genre).
      │  Enriches each new song with real audio features from the Spotify API.
      │  Inserts new songs into the catalog and generates weekly interaction data.
      ▼
  validate_data
      │  Data quality gate: blocks retraining if the database doesn't meet
      │  minimum thresholds (≥100 songs, ≥1,000 interactions).
      │  Fail fast — don't waste 5 minutes training on a broken dataset.
      ▼
  train_models
      │  Runs train.py — retrains NMF collaborative filter,
      │  content-based similarity model, and ensemble.
      ▼
  evaluate_models
      │  Verifies all model files were written fresh by this run.
      │  Catches silent failures (train.py exits 0 but writes nothing).
      ▼
  promote_models
         Saves a timestamped snapshot of new models to models/archive/.
         Keeps the last 5 snapshots, prunes older ones (rollback capability).
```

### Data Sources (Both Free)

| API | What it provides | Rate limit |
|---|---|---|
| Last.fm | Weekly chart top tracks, top tracks by genre | 5 req/s |
| Spotify | Real audio features: energy, danceability, valence, tempo | 180 req/30s |

Registration for both takes ~5 minutes and requires no credit card.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Weekly schedule | Balances model freshness vs. compute cost |
| Validate before train | Fail fast on bad data |
| Freshness check after train | Catches silent training failures |
| Timestamped archives | Enables rollback without manual intervention |
| `LocalExecutor` + PostgreSQL | Production-grade metadata store, no Celery overhead |

---

## Performance Metrics

| Metric | Value | Context |
|---|---|---|
| **Precision@10** | 15% | Industry good: >10%, excellent: >20% |
| **Training time** | ~5 minutes | Acceptable for weekly retraining |
| **Inference latency** | <100ms | Real-time threshold (feels instant) |
| **Data sparsity** | 98% | Typical for recommendation systems |
| **Cold-start coverage** | 100% | Every user gets recommendations from day 1 |
| **Database size** | ~250MB | Fits free-tier cloud storage |

---

## Key Questions & Answers

**Q: How does your system handle new users with no listening history?**

Multi-tier fallback. New users (0 interactions) get 70% content-based + 30% popularity. After 1–5 listens, weights shift to 50/50. Active users (5+ interactions) get the full 60/30/10 ensemble. Result: 100% coverage while gradually personalising.

**Q: Why is inference so fast (<100ms)?**

Four compounding optimisations: sparse matrices (98% sparsity = ~50× memory savings), pre-computed NMF embeddings (calculated at training, not inference), vectorised NumPy operations, and model-level caching (load once, serve indefinitely).

**Q: What's your biggest system bottleneck?**

Database writes. SQLite doesn't handle concurrent writes — if 100 users played songs simultaneously, you'd get write conflicts. Fix: migrate to PostgreSQL with connection pooling. At current scale (500 users), SQLite's simplicity and $0 cost outweigh the limitation.

**Q: How would you scale to 1 million users?**

1. PostgreSQL for concurrent writes and horizontal scaling
2. Redis for distributed model caching across servers
3. Separate training pipeline (weekly Airflow job) from inference API (always-on)
4. Two-tower neural networks if accuracy becomes the bottleneck at scale

Estimated cost at 1M users: $200–500/month.

**Q: Why synthetic data for the live demo?**

Technical constraints: Streamlit Cloud has a 100MB upload limit (database + models = ~450MB total). The production version is available by cloning the repository — same algorithms, real data.

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| ML | scikit-learn NMF | Industry standard, pure Python, no C++ compiler |
| Database | SQLite → PostgreSQL | Start at $0, migrate when needed |
| UI | Streamlit | Functional UI in ~30 minutes vs hours for React |
| Orchestration | Apache Airflow | Industry-standard pipeline scheduler |
| Containerisation | Docker Compose | Reproducible local environment |
| Sparse matrices | SciPy CSR | 50× memory reduction at 98% sparsity |

**Philosophy:** Choose boring technology that scales. Solve the problem in front of you, document the migration path for the problem you'll have next.

---

## ML System Design Principles Demonstrated

1. **Latency vs Accuracy Trade-offs** — NMF over deep learning (6× faster, 2–3% accuracy cost)
2. **Cold-Start Solutions** — Multi-tier fallback ensures 100% user coverage from day 1
3. **Intelligent Caching** — Cache models (expensive), not recommendations (cheap + personalised)
4. **Scalability Planning** — Migration path documented before it's needed (SQLite → PostgreSQL)
5. **Cost Optimisation** — $0 infrastructure at MVP with a clear path to $50–100/month at scale
6. **Real-Time Performance** — <100ms through sparse matrices and pre-computation
7. **Data Engineering** — Automated weekly retraining pipeline with Airflow (v1.3)
8. **Production Thinking** — Not just models, but systems that ship, scale, and self-maintain

---

## Project Evolution

This project followed a deliberate build-validate-scale approach:

1. **v1.0 — Core ML system:** Built and validated the recommendation engine using synthetic data to prove the pipeline works before adding real-world complexity
2. **v1.2 — Real data + search:** Integrated 10,000 actual songs with real audio features; added search functionality and fixed recommendation diversity
3. **v1.3 — Automated retraining (Airflow):** Added a 5-stage weekly retraining DAG backed by real Last.fm and Spotify data, with data validation, model freshness checks, and versioned archiving

This iterative approach mirrors CI/CD principles: prove it works in a controlled environment, then add production complexity incrementally.

---

*Emmanuel Osabutey — tettehosabutey@outlook.com*
