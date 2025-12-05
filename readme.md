# Music Recommendation System

A music recommendation system combining collaborative filtering, content-based filtering, and popularity signals. Uses ALS for matrix factorization with SQLite for data storage.

Python 3.8+ | MIT License

## Overview

The system uses an ensemble approach with three components:
- **ALS collaborative filtering** (60% weight) - Matrix factorization on user-song interactions
- **Content-based filtering** (30% weight) - Audio feature similarity using cosine distance
- **Popularity baseline** (10% weight) - Trending songs and play counts

Weights adapt based on user history. New users rely more heavily on content and popularity until sufficient interaction data exists.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION SYSTEM                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            ENSEMBLE RECOMMENDER                      │  │
│  │         (Adaptive weighting by user history)         │  │
│  └──────────────────────────────────────────────────────┘  │
│           ▲                  ▲                    ▲          │
│           │                  │                    │          │
│  ┌────────┴────────┐  ┌─────┴─────┐  ┌──────────┴────────┐ │
│  │   ALS Model     │  │  Content  │  │   Popularity      │ │
│  │   (60% weight)  │  │  Based    │  │   Baseline        │ │
│  │                 │  │ (30% wt)  │  │   (10% weight)    │ │
│  │  • 64 factors   │  │ • Audio   │  │  • Play counts    │ │
│  │  • Matrix fact  │  │   features│  │  • Cached daily   │ │
│  └─────────────────┘  └───────────┘  └───────────────────┘ │
│           ▲                  ▲                    ▲          │
│           │                  │                    │          │
│  ┌────────┴──────────────────┴────────────────────┴──────┐  │
│  │                DATA LAYER (SQLite)                    │  │
│  │  • Users, Songs, Interactions                        │  │
│  │  • Indexed for fast lookups                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone repository
cd music-recommendation-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data and train models
python src/train/train.py --generate-data

# Test inference
python src/models/als_recommender.py
```

## Project Structure

```
music-recommendation-system/
├── data/
│   ├── music_rec.db              # SQLite database
│   └── id_mappings.pkl           # User/song ID mappings
│
├── models/
│   ├── als_model.pkl             # Collaborative filtering model
│   ├── content_based_model.pkl   # Content-based model
│   ├── ensemble_config.pkl       # Ensemble configuration
│   └── training_results.json     # Evaluation metrics
│
├── src/
│   ├── data/
│   │   ├── data_generator.py    # Synthetic data generation
│   │   └── data_loader.py       # Data loading utilities
│   │
│   ├── models/
│   │   ├── als_recommender.py
│   │   ├── content_based_recommender.py
│   │   └── ensemble_recommender.py
│   │
│   └── train/
│       └── train.py              # Training pipeline
│
├── requirements.txt
└── README.md
```

## Technical Details

### ALS Configuration

```python
factors = 64           # Latent dimensions
iterations = 15        # Training epochs
regularization = 0.01  # L2 regularization
alpha = 40            # Confidence weighting for implicit feedback
```

**Complexity:**
- Training: O(iterations × (n_users × factors² + n_items × factors²))
- Inference: O(n_items × factors)
- Memory: ~4MB for 10K users, 5K songs

### Content-Based Features

Audio features (8 dimensions):
- acousticness [0-1]
- danceability [0-1]
- energy [0-1]
- instrumentalness [0-1]
- liveness [0-1]
- speechiness [0-1]
- valence [0-1]
- tempo [normalized BPM]

Similarity computed using cosine distance on normalized feature vectors.

### Ensemble Strategy

Weighting adjusts based on interaction history:
- **New user (0 interactions)**: Content 70% + Popular 30%
- **Cold start (<5 interactions)**: Content 50% + Popular 50%
- **Active user (≥5 interactions)**: ALS 60% + Content 30% + Popular 10%

Diversity optimization uses Maximal Marginal Relevance (MMR) with 70% relevance and 30% diversity weighting.

## Performance

### Benchmarks (10K users, 5K songs)

**Training:**
- Generation: ~2 min
- Training: ~30s
- Model size: 4MB

**Inference:**
- ALS: ~5ms
- Content-based: ~2ms
- Ensemble: ~7ms

**Metrics:**
- Precision@5: 0.18
- Precision@10: 0.15
- NDCG@10: 0.27
- Coverage: 35%

### Scaling Estimates

| Dataset | Training Time | Model Size |
|---------|--------------|------------|
| 1K users | ~3s | 1 MB |
| 10K users | ~30s | 4 MB |
| 100K users | ~5 min | 40 MB |
| 1M users | ~50 min | 400 MB |

Note: 1M+ users would require distributed training (Ray/Spark) and migration from SQLite to PostgreSQL.

## Design Decisions

### Storage: SQLite

Using SQLite for simplicity and zero cost. Handles up to ~100K users on a single machine. Migration path to PostgreSQL is straightforward when scaling needs arise.

### Caching

In-memory Python dictionaries for:
- Popular songs (updated daily)
- User ID mappings
- Precomputed similarities

Cache clears on restart, which is acceptable for development and small-scale deployments.

### Evaluation Metrics

- **Precision@K**: Measures relevance of top K recommendations
- **NDCG@K**: Measures ranking quality (position-aware)
- **Coverage**: Catalog diversity metric

Using these over AUC/RMSE because we're optimizing for ranking with implicit feedback, not classification or rating prediction.

### Train/Test Split

Currently using random 80/20 split for balanced evaluation. Temporal split would be more appropriate for production (train on past, test on future).

## Usage Examples

### Basic Recommendations

```python
from src.models.ensemble_recommender import EnsembleRecommender

ensemble = EnsembleRecommender.load('models/')
recommendations = ensemble.recommend(user_id='user_00001', top_k=10)
```

### Cold Start (New User)

```python
# User with only a few likes
recommendations = ensemble.recommend(
    user_id=None,
    liked_song_ids=['song_00042', 'song_00103'],
    top_k=10
)
```

### Similar Songs

```python
from src.models.als_recommender import ALSRecommender

als = ALSRecommender.load('models/als_model.pkl')
similar = als.get_similar_songs('song_00042', top_k=5)
```

### Explainability

```python
explanation = ensemble.explain_recommendation('song_00042', user_id='user_00001')
# Returns breakdown of why this song was recommended
```

## Model Selection

Chose ALS over alternatives for the initial implementation:

| Method | Training | Inference | Complexity |
|--------|----------|-----------|------------|
| ALS | Fast | Fast | Medium |
| SGD Matrix Factorization | Medium | Fast | Medium |
| Neural Collaborative Filtering | Slow | Medium | High |
| Transformer-based | Very Slow | Slow | Very High |

ALS provides good accuracy with minimal complexity and infrastructure requirements. Two-tower networks or transformers would be worth exploring at larger scale.

## Future Improvements

- Hyperparameter tuning (grid search for ALS parameters)
- Additional features (artist embeddings, lyrics sentiment)
- Personalized ensemble weights per user segment
- Temporal split evaluation
- API layer with FastAPI
- Containerization with Docker
- Monitoring and logging

## Requirements

See `requirements.txt` for full dependency list. Main dependencies:
- implicit (ALS implementation)
- numpy, scipy
- scikit-learn
- pandas
- sqlite3 (standard library)

## License

MIT