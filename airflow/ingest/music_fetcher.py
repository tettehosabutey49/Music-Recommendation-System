"""
Real Music Data Fetcher
=======================

Fetches this week's top tracks from Last.fm, enriches each new track with
real audio features from the Spotify API, then upserts into the local SQLite DB.

Setup (one-time, both are free — no credit card needed):
    1. Last.fm API key:  https://www.last.fm/api/account/create
    2. Spotify app:      https://developer.spotify.com/dashboard

    Then set in airflow/.env (or as environment variables):
        LASTFM_API_KEY=your_lastfm_key
        SPOTIFY_CLIENT_ID=your_spotify_client_id
        SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

How it works:
    1. Fetch Last.fm's weekly chart top tracks (global trending)
    2. Fetch top tracks per genre using Last.fm tags
    3. For each track not already in the DB:
       a. Search Spotify for the track ID
       b. Pull real audio features: energy, danceability, valence, tempo
       c. Fall back to genre-informed random values if Spotify can't find it
    4. Insert new songs into the songs table
    5. Generate simulated play interactions so new songs enter the recommendation pool
    6. Add new weekly listening interactions for existing songs (simulates user activity)
"""

from __future__ import annotations

import hashlib
import os
import random
import sqlite3
import time
from datetime import datetime
from typing import Optional

import requests

# ── API endpoints ──────────────────────────────────────────────────────────────

LASTFM_BASE = "http://ws.audioscrobbler.com/2.0/"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_BASE = "https://api.spotify.com/v1"

# Last.fm tag names → your DB genre labels
GENRE_TAGS: dict[str, str] = {
    "Pop": "pop",
    "Hip-Hop": "hip-hop",
    "Rock": "rock",
    "Electronic": "electronic dance music",
    "R&B": "rnb",
    "Jazz": "jazz",
    "Country": "country",
    "Latin": "latin",
    "Metal": "metal",
    "Indie": "indie",
}

# ── Last.fm helpers ────────────────────────────────────────────────────────────

def get_lastfm_chart_tracks(api_key: str, limit: int = 50) -> list[tuple[str, str, str]]:
    """Return (title, artist, genre) tuples for this week's global chart."""
    params = {
        "method": "chart.getTopTracks",
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    resp = requests.get(LASTFM_BASE, params=params, timeout=10)
    resp.raise_for_status()
    tracks = resp.json().get("tracks", {}).get("track", [])
    return [
        (t.get("name", ""), t.get("artist", {}).get("name", ""), "Pop")
        for t in tracks
        if t.get("name") and t.get("artist", {}).get("name")
    ]


def get_lastfm_genre_tracks(
    api_key: str, tag: str, genre_label: str, limit: int = 20
) -> list[tuple[str, str, str]]:
    """Return (title, artist, genre) tuples for a Last.fm genre tag."""
    params = {
        "method": "tag.getTopTracks",
        "tag": tag,
        "api_key": api_key,
        "format": "json",
        "limit": limit,
    }
    resp = requests.get(LASTFM_BASE, params=params, timeout=10)
    resp.raise_for_status()
    tracks = resp.json().get("tracks", {}).get("track", [])
    return [
        (t.get("name", ""), t.get("artist", {}).get("name", ""), genre_label)
        for t in tracks
        if t.get("name") and t.get("artist", {}).get("name")
    ]


# ── Spotify helpers ────────────────────────────────────────────────────────────

def get_spotify_token(client_id: str, client_secret: str) -> str:
    """Client credentials flow — no user login needed."""
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_spotify_track_metadata(token: str, title: str, artist: str) -> Optional[dict]:
    """
    Search Spotify for a track and return useful metadata.

    Note: The audio-features endpoint was deprecated for new apps in Nov 2024
    and now returns 403. We use the search endpoint instead, which provides
    duration, album name, and popularity — all useful for enriching songs.

    Audio features (energy, danceability, valence, tempo) fall back to
    genre-informed defaults in _genre_audio_defaults().
    """
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": f"track:{title} artist:{artist}",
        "type": "track",
        "limit": 1,
    }
    resp = requests.get(
        f"{SPOTIFY_BASE}/search", headers=headers, params=params, timeout=10
    )
    if resp.status_code != 200:
        return None
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    t = items[0]
    return {
        "duration_ms":  t.get("duration_ms", 210_000),
        "album":        t.get("album", {}).get("name", ""),
        "popularity":   t.get("popularity", 50),   # 0-100, used to weight initial interactions
    }


# ── Core utilities ─────────────────────────────────────────────────────────────

def make_song_id(title: str, artist: str) -> str:
    """
    Generate a stable, deterministic song_id from title + artist.
    Using a hash means re-running the fetcher won't create duplicate rows
    for the same song.
    """
    key = f"{title.lower().strip()}|{artist.lower().strip()}"
    return "lastfm_" + hashlib.md5(key.encode()).hexdigest()[:12]


def _genre_audio_defaults(genre: str) -> dict:
    """
    Fallback audio features when Spotify can't find a track.
    Values are genre-informed so new songs still land in roughly the right
    part of the feature space for content-based recommendations.
    """
    defaults = {
        "Pop":        {"energy": (0.5, 0.85), "danceability": (0.55, 0.9), "valence": (0.4, 0.85), "tempo": (100, 135)},
        "Hip-Hop":    {"energy": (0.5, 0.85), "danceability": (0.65, 0.95), "valence": (0.3, 0.75), "tempo": (75, 110)},
        "Rock":       {"energy": (0.6, 0.95), "danceability": (0.3, 0.65), "valence": (0.3, 0.7), "tempo": (110, 160)},
        "Electronic": {"energy": (0.65, 0.95), "danceability": (0.65, 0.95), "valence": (0.35, 0.75), "tempo": (120, 145)},
        "R&B":        {"energy": (0.35, 0.7), "danceability": (0.55, 0.85), "valence": (0.3, 0.75), "tempo": (70, 110)},
        "Jazz":       {"energy": (0.2, 0.6), "danceability": (0.3, 0.65), "valence": (0.3, 0.75), "tempo": (80, 180)},
        "Country":    {"energy": (0.4, 0.8), "danceability": (0.4, 0.7), "valence": (0.45, 0.85), "tempo": (90, 140)},
        "Latin":      {"energy": (0.55, 0.9), "danceability": (0.65, 0.95), "valence": (0.5, 0.9), "tempo": (90, 130)},
        "Metal":      {"energy": (0.75, 0.99), "danceability": (0.15, 0.5), "valence": (0.1, 0.5), "tempo": (130, 200)},
        "Indie":      {"energy": (0.3, 0.75), "danceability": (0.35, 0.7), "valence": (0.25, 0.7), "tempo": (90, 140)},
    }
    d = defaults.get(genre, {"energy": (0.3, 0.9), "danceability": (0.3, 0.9), "valence": (0.2, 0.8), "tempo": (80, 160)})
    return {
        "energy":       round(random.uniform(*d["energy"]), 3),
        "danceability": round(random.uniform(*d["danceability"]), 3),
        "valence":      round(random.uniform(*d["valence"]), 3),
        "tempo":        round(random.uniform(*d["tempo"]), 1),
        "duration":     random.randint(180, 270),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def fetch_and_upsert(
    db_path: str,
    lastfm_api_key: str,
    spotify_client_id: str,
    spotify_client_secret: str,
    tracks_per_genre: int = 20,
    weekly_interaction_count: int = 3_000,
) -> dict:
    """
    Full ingestion run:
      1. Fetch top tracks from Last.fm (chart + per genre)
      2. Enrich new tracks with Spotify audio features
      3. Upsert into songs table
      4. Generate weekly interaction data for the recommendation model

    Returns a summary dict with counts for logging / XCom.
    """
    print("[music_fetcher] Starting weekly music ingest...")

    # ── Spotify token (valid for 1 hour — enough for one run) ─────────────────
    spotify_token = get_spotify_token(spotify_client_id, spotify_client_secret)
    print("[music_fetcher] Spotify token acquired.")

    conn = sqlite3.connect(db_path)

    existing_ids: set[str] = {
        r[0] for r in conn.execute("SELECT song_id FROM songs").fetchall()
    }
    user_ids: list[str] = [
        r[0] for r in conn.execute("SELECT user_id FROM users").fetchall()
    ]
    if not user_ids:
        conn.close()
        raise ValueError("No users in database — run fetch_real_music_data.py first.")

    # ── 1. Collect tracks from Last.fm ────────────────────────────────────────
    all_tracks: list[tuple[str, str, str]] = []

    print("[music_fetcher] Fetching Last.fm chart top tracks...")
    try:
        chart = get_lastfm_chart_tracks(lastfm_api_key, limit=50)
        all_tracks.extend(chart)
        print(f"  Got {len(chart)} chart tracks.")
    except Exception as e:
        print(f"  [warning] Chart fetch failed: {e}")

    for genre_label, tag in GENRE_TAGS.items():
        try:
            genre_tracks = get_lastfm_genre_tracks(
                lastfm_api_key, tag, genre_label, limit=tracks_per_genre
            )
            all_tracks.extend(genre_tracks)
            time.sleep(0.25)   # stay within Last.fm's 5 req/s limit
        except Exception as e:
            print(f"  [warning] Genre '{genre_label}' fetch failed: {e}")

    print(f"[music_fetcher] {len(all_tracks)} tracks fetched. Filtering new ones...")

    # Deduplicate within this batch
    seen: set[str] = set()
    unique_tracks = []
    for title, artist, genre in all_tracks:
        sid = make_song_id(title, artist)
        if sid not in existing_ids and sid not in seen:
            seen.add(sid)
            unique_tracks.append((title, artist, genre, sid))

    print(f"[music_fetcher] {len(unique_tracks)} new tracks to enrich and insert.")

    # ── 2. Enrich with Spotify + insert into songs table ─────────────────────
    new_songs = 0
    new_song_ids: list[str] = []

    for title, artist, genre, song_id in unique_tracks:
        # Spotify search: duration, album, popularity (audio features API deprecated Nov 2024)
        spotify_meta = get_spotify_track_metadata(spotify_token, title, artist)
        time.sleep(0.15)   # gentle on Spotify rate limits

        # Genre-informed audio features (energy, danceability, valence, tempo)
        af = _genre_audio_defaults(genre)
        energy       = af["energy"]
        danceability = af["danceability"]
        valence      = af["valence"]
        tempo        = af["tempo"]

        if spotify_meta:
            duration   = int(spotify_meta["duration_ms"] / 1000)
            album      = spotify_meta["album"]
            popularity = spotify_meta["popularity"]   # 0-100
            source     = "spotify+genre-defaults"
        else:
            duration   = af["duration"]
            album      = ""
            popularity = 50
            source     = "genre-defaults"

        conn.execute(
            """INSERT OR IGNORE INTO songs
               (song_id, title, artist, genre, year, duration, energy, danceability, valence, tempo, album)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (song_id, title, artist, genre, datetime.now().year,
             duration, energy, danceability, valence, tempo, album),
        )
        existing_ids.add(song_id)
        new_song_ids.append((song_id, popularity))
        new_songs += 1
        print(f"  + [{source}] {title} — {artist} ({genre})")

    # ── 3. Seed interactions for new songs ────────────────────────────────────
    # Give each new song initial play data so it enters the recommendation pool.
    # We weight play counts by Spotify popularity so genuinely popular songs
    # get more initial signal than obscure ones.
    new_interaction_rows = []
    for sid, popularity in new_song_ids:
        sample_users = random.sample(user_ids, min(15, len(user_ids)))
        for uid in sample_users:
            # popularity 0-100 → play_count roughly 1-10
            play_count = max(1, round(popularity / 15) + random.randint(0, 2))
            new_interaction_rows.append((
                uid, sid, play_count, datetime.now().isoformat(),
            ))

    # ── 4. Weekly listening interactions for existing songs ───────────────────
    # Simulates user activity during the past week so the model learns from
    # recent listening patterns, not just historical data.
    existing_song_ids = list(existing_ids)
    for _ in range(weekly_interaction_count):
        new_interaction_rows.append((
            random.choice(user_ids),
            random.choice(existing_song_ids),
            random.randint(1, 8),
            datetime.now().isoformat(),
        ))

    conn.executemany(
        "INSERT INTO interactions (user_id, song_id, play_count, last_played) VALUES (?, ?, ?, ?)",
        new_interaction_rows,
    )
    conn.commit()
    conn.close()

    total_interactions = len(new_interaction_rows)
    print(
        f"[music_fetcher] Done. "
        f"New songs: {new_songs} | New interactions: {total_interactions}"
    )
    return {"new_songs": new_songs, "new_interactions": total_interactions}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "data/music_rec.db"
    result = fetch_and_upsert(
        db_path=db,
        lastfm_api_key=os.environ["LASTFM_API_KEY"],
        spotify_client_id=os.environ["SPOTIFY_CLIENT_ID"],
        spotify_client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    )
    print(result)
