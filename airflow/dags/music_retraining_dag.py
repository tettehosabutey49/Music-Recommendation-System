"""
Music Recommendation System - Weekly Retraining DAG
====================================================

Automates the full model retraining pipeline on a weekly schedule.

Pipeline stages:
    ingest_data  →  validate_data  →  train_models  →  evaluate_models  →  promote_models

Where does new data come from?
    Real music data! Each run calls:
      - Last.fm chart.getTopTracks  → this week's globally trending songs
      - Last.fm tag.getTopTracks    → top songs per genre (Pop, Rock, Hip-Hop, etc.)
      - Spotify audio-features API  → real energy, danceability, valence, tempo

    New songs are inserted into the songs table and seeded with initial
    interactions so they enter the recommendation pool immediately.
    Existing songs get new weekly listening interactions so the model
    learns from recent behaviour, not just historical data.

    API keys required (both free, no credit card):
      LASTFM_API_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
      → Set these in airflow/.env

To run locally:
    cd airflow && docker compose up -d
    Open http://localhost:8080 (admin / admin), then enable this DAG.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("MUSIC_REC_ROOT", "/opt/airflow/project")
DATA_DB = f"{PROJECT_ROOT}/data/music_rec.db"
MODELS_DIR = f"{PROJECT_ROOT}/models"
ARCHIVE_DIR = f"{PROJECT_ROOT}/models/archive"

# Minimum data thresholds before allowing retraining
MIN_SONGS = 100
MIN_INTERACTIONS = 1_000
MAX_ARCHIVE_VERSIONS = 5      # Keep last N model snapshots

DEFAULT_ARGS = {
    "owner": "music-rec",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── Task: Data Ingestion ───────────────────────────────────────────────────────

def ingest_data(**context):
    """
    Fetch this week's real music data from Last.fm + Spotify and load it
    into the local SQLite database.

    Requires three environment variables (set in airflow/.env):
        LASTFM_API_KEY        — free at https://www.last.fm/api/account/create
        SPOTIFY_CLIENT_ID     — free at https://developer.spotify.com/dashboard
        SPOTIFY_CLIENT_SECRET

    Falls back to genre-informed synthetic audio features for any tracks
    that Spotify can't match, so the task never fails due to API gaps.

    Pushes a summary dict to XCom for downstream tasks to log.
    """
    import sys
    import os

    # Make the ingest module importable from inside the Airflow container
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "airflow"))

    from ingest.music_fetcher import fetch_and_upsert

    lastfm_key       = os.environ.get("LASTFM_API_KEY", "")
    spotify_id       = os.environ.get("SPOTIFY_CLIENT_ID", "")
    spotify_secret   = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    if not all([lastfm_key, spotify_id, spotify_secret]):
        raise EnvironmentError(
            "Missing API credentials. Set LASTFM_API_KEY, SPOTIFY_CLIENT_ID, "
            "and SPOTIFY_CLIENT_SECRET in airflow/.env"
        )

    result = fetch_and_upsert(
        db_path=DATA_DB,
        lastfm_api_key=lastfm_key,
        spotify_client_id=spotify_id,
        spotify_client_secret=spotify_secret,
    )

    print(f"[ingest_data] New songs: {result['new_songs']} | New interactions: {result['new_interactions']}")
    context["ti"].xcom_push(key="ingest_summary", value=result)


# ── Task: Data Validation ──────────────────────────────────────────────────────

def validate_data(**context):
    """
    Verify the SQLite database exists and meets minimum thresholds before
    triggering an expensive retraining run.

    Pushes record counts to XCom so downstream tasks can log them.
    """
    db_path = Path(DATA_DB)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {DATA_DB}. "
            "Run fetch_real_music_data.py or train.py --generate-data first."
        )

    conn = sqlite3.connect(DATA_DB)
    try:
        songs        = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        users        = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        interactions = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    finally:
        conn.close()

    print(f"[validate_data] songs={songs}, users={users}, interactions={interactions}")

    if songs < MIN_SONGS:
        raise ValueError(
            f"Only {songs} songs in DB (minimum {MIN_SONGS}). "
            "Add more data before retraining."
        )
    if interactions < MIN_INTERACTIONS:
        raise ValueError(
            f"Only {interactions} interactions (minimum {MIN_INTERACTIONS}). "
            "Not enough signal to retrain reliably."
        )

    # Push counts so the evaluate task can log them in context
    context["ti"].xcom_push(
        key="record_counts",
        value={"songs": songs, "users": users, "interactions": interactions},
    )
    print("[validate_data] Data validation passed.")


# ── Task: Model Evaluation ─────────────────────────────────────────────────────

def evaluate_models(**context):
    """
    After training, verify that all expected model files exist and were
    written during this run (not stale from a previous run).

    Raises if any file is missing or was not freshly written.
    """
    required_files = [
        "als_model.pkl",
        "content_based_model.pkl",
        "ensemble_config.pkl",
    ]
    models_path = Path(MODELS_DIR)
    freshness_threshold_minutes = 120   # training should finish within 2 hours

    # Pull record counts from XCom for contextual logging
    counts = context["ti"].xcom_pull(
        task_ids="validate_data", key="record_counts"
    ) or {}
    print(f"[evaluate_models] Trained on: {counts}")

    for fname in required_files:
        fpath = models_path / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected model file not found after training: {fpath}"
            )
        age_min = (datetime.now() - datetime.fromtimestamp(fpath.stat().st_mtime)).total_seconds() / 60
        print(f"  {fname}: {age_min:.1f} min old")
        if age_min > freshness_threshold_minutes:
            raise RuntimeError(
                f"{fname} was not updated by this run "
                f"(last modified {age_min:.0f} min ago, threshold {freshness_threshold_minutes} min). "
                "Training may have failed silently."
            )

    print("[evaluate_models] All model files validated.")


# ── Task: Model Promotion ──────────────────────────────────────────────────────

def promote_models(**context):
    """
    Archive a timestamped snapshot of the newly trained models, then prune
    old snapshots keeping only the last MAX_ARCHIVE_VERSIONS.

    The live models remain in models/ — the archive is a rollback safety net.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = Path(ARCHIVE_DIR) / ts
    archive_path.mkdir(parents=True, exist_ok=True)

    model_files = [
        "als_model.pkl",
        "als_mappings.pkl",
        "content_based_model.pkl",
        "ensemble_config.pkl",
    ]

    for fname in model_files:
        src = Path(MODELS_DIR) / fname
        if src.exists():
            shutil.copy2(src, archive_path / fname)
            print(f"  Archived: {fname}")

    print(f"[promote_models] Snapshot saved to: {archive_path}")

    # Prune old archives — keep the most recent MAX_ARCHIVE_VERSIONS
    archive_root = Path(ARCHIVE_DIR)
    all_snapshots = sorted(archive_root.glob("*/"), reverse=True)
    for old_snapshot in all_snapshots[MAX_ARCHIVE_VERSIONS:]:
        shutil.rmtree(old_snapshot, ignore_errors=True)
        print(f"  Pruned old archive: {old_snapshot.name}")

    print(f"[promote_models] Models promoted. Archive count capped at {MAX_ARCHIVE_VERSIONS}.")


# ── DAG Definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="music_retraining_weekly",
    description="Weekly automated retraining for the music recommendation system",
    schedule_interval="@weekly",      # Runs every Sunday at 00:00 UTC
    start_date=days_ago(1),
    catchup=False,                    # Don't backfill missed runs
    default_args=DEFAULT_ARGS,
    tags=["music-rec", "ml", "retraining"],
) as dag:

    ingest_data_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
        doc_md=(
            "Pull in new user listening data for the week. "
            "In production: replace with your event source (S3, Kafka, API). "
            "Here: generates synthetic interactions as a realistic stand-in."
        ),
    )

    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        doc_md="Check the database exists and meets minimum record thresholds.",
    )

    train_models_task = BashOperator(
        task_id="train_models",
        bash_command=f"cd {PROJECT_ROOT} && python train.py",
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        doc_md="Run the full training pipeline (ALS + content-based + ensemble).",
    )

    evaluate_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        doc_md="Verify all model files were written fresh by this run.",
    )

    promote_models_task = PythonOperator(
        task_id="promote_models",
        python_callable=promote_models,
        doc_md="Archive a versioned snapshot of the new models, prune old ones.",
    )

    # Pipeline dependency chain
    ingest_data_task >> validate_data_task >> train_models_task >> evaluate_models_task >> promote_models_task
