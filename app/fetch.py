from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import spotipy
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import ListeningHistory, Track, User

logger = logging.getLogger(__name__)

_artist_genre_cache: dict[str, str] = {}

LASTFM_BLOCKLIST: frozenset[str] = frozenset([
    "seen live", "favourite", "favorites", "love", "loved", "my favourite",
    "awesome", "great", "good", "best", "amazing", "cool", "nice", "perfect",
    "beautiful", "wonderful", "excellent", "brilliant",
])
_LASTFM_COUNT_THRESHOLD = 25
_LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
_lastfm_cache: dict[tuple, list[str]] = {}


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _enrich_genres_batch(
    sp: spotipy.Spotify,
    session: Session,
    artist_map: dict[str, list[int]],
) -> None:
    """
    Batch-fetch genres for all artists in artist_map and update their tracks.

    artist_map: {spotify_artist_id -> [track.id, ...]}
    Uses sp.artists() (max 50 IDs per call) instead of one sp.artist() call per
    artist, reducing N calls to ceil(N/50) calls.
    """
    unknown = [aid for aid in artist_map if aid not in _artist_genre_cache]
    for i in range(0, len(unknown), 50):
        batch = unknown[i : i + 50]
        time.sleep(0.1)
        try:
            result = sp.artists(batch)
            for artist_data in result.get("artists") or []:
                if artist_data:
                    _artist_genre_cache[artist_data["id"]] = ", ".join(
                        artist_data.get("genres", [])
                    )
        except Exception as exc:
            logger.warning("Batch artist genres failed (batch %d): %s", i // 50, exc)
    # Mark any artist the API didn't return so we don't retry on next sync
    for aid in unknown:
        _artist_genre_cache.setdefault(aid, "")

    for artist_id, track_ids in artist_map.items():
        genres_str = _artist_genre_cache.get(artist_id, "")
        for track_id in track_ids:
            track = session.get(Track, track_id)
            if track is not None and track.genres is None:
                track.genres = genres_str
    session.commit()


def load_lastfm_api_key() -> str:
    """Read LASTFM_API_KEY from environment; raise RuntimeError if missing."""
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        raise RuntimeError(
            "LASTFM_API_KEY is not set. Add it to your .env file."
        )
    return key


def _filter_lastfm_tags(tags_raw: list[dict]) -> list[str]:
    """Apply blocklist and count threshold to a raw Last.fm tag list."""
    result = []
    for t in tags_raw:
        try:
            count = int(t.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        name = t.get("name", "").lower()
        if count >= _LASTFM_COUNT_THRESHOLD and name not in LASTFM_BLOCKLIST:
            result.append(name)
    return result


def _fetch_lastfm_track_tags(artist: str, track_name: str, api_key: str) -> list[str]:
    """Return filtered Last.fm top tags for a specific track (cached per artist+track)."""
    cache_key = (artist, track_name)
    if cache_key in _lastfm_cache:
        return _lastfm_cache[cache_key]
    time.sleep(0.2)
    try:
        resp = requests.get(
            _LASTFM_BASE_URL,
            params={
                "method": "track.getTopTags",
                "artist": artist,
                "track": track_name,
                "api_key": api_key,
                "format": "json",
            },
            timeout=10,
        )
        tags = _filter_lastfm_tags(resp.json().get("toptags", {}).get("tag", []))
    except Exception as exc:
        logger.warning("Last.fm track tags failed ('%s' / '%s'): %s", artist, track_name, exc)
        tags = []
    _lastfm_cache[cache_key] = tags
    return tags


def _fetch_lastfm_artist_tags(artist: str, api_key: str) -> list[str]:
    """Return filtered Last.fm top tags for an artist (cached per artist)."""
    cache_key = (artist,)
    if cache_key in _lastfm_cache:
        return _lastfm_cache[cache_key]
    time.sleep(0.2)
    try:
        resp = requests.get(
            _LASTFM_BASE_URL,
            params={
                "method": "artist.getTopTags",
                "artist": artist,
                "api_key": api_key,
                "format": "json",
            },
            timeout=10,
        )
        tags = _filter_lastfm_tags(resp.json().get("toptags", {}).get("tag", []))
    except Exception as exc:
        logger.warning("Last.fm artist tags failed ('%s'): %s", artist, exc)
        tags = []
    _lastfm_cache[cache_key] = tags
    return tags


def fetch_lastfm_tags(session: Session, user: User, api_key: str) -> int:
    """
    Fetch Last.fm tags for all tracks in the user's library missing them.

    Fallback order per track:
      1. track.getTopTags
      2. artist.getTopTags (primary artist only; memoized across tracks by same artist)
      3. existing genres column (split + lowercase)
      4. "" stored to prevent retry on next sync

    Returns count of tracks updated.
    """
    tracks = (
        session.query(Track)
        .filter_by(user_id=user.id)
        .filter(Track.lastfm_tags.is_(None))
        .all()
    )
    count = 0
    for track in tracks:
        name = track.name or ""
        # Use primary artist only — Last.fm doesn't support multi-artist queries
        raw_artist = track.artist or ""
        primary_artist = raw_artist.split(",")[0].strip()

        tags = _fetch_lastfm_track_tags(primary_artist, name, api_key)

        if not tags:
            tags = _fetch_lastfm_artist_tags(primary_artist, api_key)

        if not tags and track.genres:
            tags = [g.strip().lower() for g in track.genres.split(",") if g.strip()]

        track.lastfm_tags = ", ".join(tags)
        count += 1

    session.commit()
    logger.info("Last.fm tags fetched for %d tracks", count)
    return count


def upsert_user(session: Session, sp: spotipy.Spotify) -> User:
    """Fetch the current Spotify profile and upsert into the users table."""
    profile = sp.current_user()
    user = session.query(User).filter_by(spotify_id=profile["id"]).first()
    if not user:
        user = User(spotify_id=profile["id"])
        session.add(user)
    user.display_name = profile.get("display_name")
    user.email = profile.get("email") or ""
    session.commit()
    return user


def _upsert_track(
    session: Session,
    user_id: int,
    item: dict,
    added_at: Optional[str] = None,
) -> tuple[Track, Optional[str]]:
    """
    Persist a track from a Spotify API item dict.
    Handles both bare track objects and wrapper dicts with a "track" key.
    Returns (track, primary_artist_id) where primary_artist_id is set only when
    genres are still missing — caller must pass it to _enrich_genres_batch.
    """
    track_data: dict = item.get("track") or item
    spotify_id: str = track_data["id"]

    track = (
        session.query(Track)
        .filter_by(user_id=user_id, spotify_track_id=spotify_id)
        .first()
    )
    if not track:
        artists = track_data.get("artists", [])
        track = Track(
            user_id=user_id,
            spotify_track_id=spotify_id,
            name=track_data["name"],
            artist=", ".join(a["name"] for a in artists),
            album=(track_data.get("album") or {}).get("name"),
            duration_ms=track_data.get("duration_ms"),
            popularity=track_data.get("popularity"),
            added_at=_parse_iso(added_at),
        )
        session.add(track)
        session.flush()

    if track.genres is None:
        artists = track_data.get("artists", [])
        primary_artist_id = artists[0].get("id") if artists else None
        return track, primary_artist_id

    return track, None


def fetch_saved_tracks(sp: spotipy.Spotify, session: Session, user: User) -> int:
    """Page through the user's Liked Songs and persist all tracks."""
    count = 0
    artist_map: dict[str, list[int]] = {}
    results = sp.current_user_saved_tracks(limit=50)
    while results:
        for item in results["items"]:
            track, artist_id = _upsert_track(session, user.id, item, added_at=item.get("added_at"))
            if artist_id:
                artist_map.setdefault(artist_id, []).append(track.id)
            count += 1
        session.commit()
        if results.get("next"):
            time.sleep(0.1)
            results = sp.next(results)
        else:
            results = None
    _enrich_genres_batch(sp, session, artist_map)
    logger.info("Saved tracks synced: %d", count)
    return count


def fetch_top_tracks(sp: spotipy.Spotify, session: Session, user: User) -> int:
    """Fetch top tracks across all three Spotify time ranges and persist them."""
    count = 0
    artist_map: dict[str, list[int]] = {}
    for time_range in ("short_term", "medium_term", "long_term"):
        results = sp.current_user_top_tracks(limit=50, time_range=time_range)
        while results:
            for item in results["items"]:
                track, artist_id = _upsert_track(session, user.id, {"track": item})
                if artist_id:
                    artist_map.setdefault(artist_id, []).append(track.id)
                count += 1
            session.commit()
            if results.get("next"):
                time.sleep(0.1)
                results = sp.next(results)
            else:
                results = None
    _enrich_genres_batch(sp, session, artist_map)
    logger.info("Top tracks synced: %d", count)
    return count


def fetch_recently_played(sp: spotipy.Spotify, session: Session, user: User) -> int:
    """Fetch up to 50 recent plays (Spotify API limit) with timestamps."""
    results = sp.current_user_recently_played(limit=50)
    count = 0
    artist_map: dict[str, list[int]] = {}
    for item in results.get("items", []):
        track, artist_id = _upsert_track(session, user.id, item)
        if artist_id:
            artist_map.setdefault(artist_id, []).append(track.id)
        played_at = _parse_iso(item.get("played_at"))
        if played_at:
            stmt = (
                sqlite_insert(ListeningHistory)
                .values(user_id=user.id, track_id=track.id, played_at=played_at)
                .on_conflict_do_nothing()
            )
            session.execute(stmt)
            count += 1
    session.commit()
    _enrich_genres_batch(sp, session, artist_map)
    logger.info("Recent plays synced: %d", count)
    return count


def load_tracks_dataframe(session: Session, user: User) -> pd.DataFrame:
    """Return a DataFrame of all tracks for a user (no audio features join required)."""
    rows = session.query(Track).filter_by(user_id=user.id).all()
    records = [
        {
            "track_id": t.id,
            "spotify_track_id": t.spotify_track_id,
            "name": t.name,
            "artist": t.artist,
            "album": t.album,
            "popularity": t.popularity or 0,
            "genres": t.genres or "",
            "lastfm_tags": t.lastfm_tags or "",
        }
        for t in rows
    ]
    return pd.DataFrame(records)


def load_history_dataframe(session: Session, user: User) -> pd.DataFrame:
    """Return a DataFrame of listening history with hour-of-day for mood inference."""
    rows = (
        session.query(ListeningHistory, Track)
        .join(Track, ListeningHistory.track_id == Track.id)
        .filter(ListeningHistory.user_id == user.id)
        .order_by(ListeningHistory.played_at)
        .all()
    )
    records = [
        {
            "played_at": lh.played_at,
            "hour": lh.played_at.hour,
            "track_id": t.id,
            "spotify_track_id": t.spotify_track_id,
            "name": t.name,
            "artist": t.artist,
        }
        for lh, t in rows
    ]
    return pd.DataFrame(records)
