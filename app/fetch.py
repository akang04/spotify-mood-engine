from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests
import spotipy
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import ListeningHistory, ScrobbleHistory, Track, User

logger = logging.getLogger(__name__)


_artist_genre_cache: dict[str, str] = {}
_genres_forbidden: bool = False


def reset_genres_forbidden() -> None:
    global _genres_forbidden
    _genres_forbidden = False


def is_genres_forbidden() -> bool:
    return _genres_forbidden


def reset_lastfm_tags(session: Session, user: User) -> None:
    """Null out all cached Last.fm tags so next sync re-fetches them."""
    session.query(Track).filter_by(user_id=user.id).update({"lastfm_tags": None})
    session.commit()

LASTFM_BLOCKLIST: frozenset[str] = frozenset([
    "seen live", "favourite", "favorites", "love", "loved", "my favourite",
    "awesome", "great", "good", "best", "amazing", "cool", "nice", "perfect",
    "beautiful", "wonderful", "excellent", "brilliant",
])
_LASTFM_COUNT_THRESHOLD = 5
_LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
_lastfm_cache: dict[tuple, list[str]] = {}
_lastfm_lock = threading.Lock()
_LASTFM_WORKERS = 5
_MAX_TOP_TRACKS_PAGES = 20
_LASTFM_MIN_INTERVAL: float = 1.0 / 5  # 5 req/s ceiling
_lastfm_rate_lock = threading.Lock()
_lastfm_last_call: float = 0.0
_LASTFM_MAX_RETRIES = 3
_LASTFM_RETRY_BACKOFF = 0.5  # seconds; doubles each attempt


def _lastfm_rate_limit() -> None:
    """Block the calling thread until it's safe to fire the next Last.fm request (≤5 req/s)."""
    global _lastfm_last_call
    while True:
        with _lastfm_rate_lock:
            now = time.monotonic()
            wait = _LASTFM_MIN_INTERVAL - (now - _lastfm_last_call)
            if wait <= 0:
                _lastfm_last_call = now
                return
        time.sleep(wait)


def _lastfm_get(params: dict) -> dict:
    """Rate-limited GET to Last.fm with retry on transient SSL/connection errors."""
    last_exc: Exception | None = None
    for attempt in range(_LASTFM_MAX_RETRIES):
        _lastfm_rate_limit()
        try:
            return requests.get(_LASTFM_BASE_URL, params=params, timeout=10).json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < _LASTFM_MAX_RETRIES - 1:
                time.sleep(_LASTFM_RETRY_BACKOFF * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


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
    global _genres_forbidden
    if _genres_forbidden:
        return
    unknown = [aid for aid in artist_map if aid not in _artist_genre_cache]
    total_batches = (len(unknown) + 49) // 50
    if unknown:
        logger.info(f"  Genre enrichment: {len(unknown)} new artists across {total_batches} batch(es)")
    for i in range(0, len(unknown), 50):
        batch = unknown[i : i + 50]
        batch_num = i // 50 + 1
        logger.info(f"    Batch {batch_num}/{total_batches}: fetching genres for {len(batch)} artists")
        time.sleep(0.1)
        try:
            result = sp.artists(batch)
            found = 0
            for artist_data in result.get("artists") or []:
                if artist_data:
                    _artist_genre_cache[artist_data["id"]] = ", ".join(
                        artist_data.get("genres", [])
                    )
                    found += 1
            logger.info(f"    Batch {batch_num}/{total_batches}: got genres for {found}/{len(batch)} artists")
        except spotipy.exceptions.SpotifyException as exc:
            if exc.http_status == 403:
                _genres_forbidden = True
                logger.info(f"  Genre enrichment blocked (403) — skipping remaining batches")
                logger.warning(
                    "Artist genre lookup forbidden (403) — Spotify app may be rate-limited. "
                    "Skipping all genre batches; Last.fm tags will be used instead."
                )
                break
            logger.warning("Batch artist genres failed (batch %d): %s", i // 50, exc)
            logger.info(f"    Batch {batch_num}/{total_batches}: FAILED — {exc}")
        except Exception as exc:
            logger.warning("Batch artist genres failed (batch %d): %s", i // 50, exc)
            logger.info(f"    Batch {batch_num}/{total_batches}: FAILED — {exc}")
    # Mark any artist the API didn't return so we don't retry on next sync
    for aid in unknown:
        _artist_genre_cache.setdefault(aid, "")
    if unknown:
        logger.info(f"  Genre enrichment done")

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


def _fetch_lastfm_track_tags(artist: str, track_name: str, api_key: str) -> list[str] | None:
    """Return filtered Last.fm top tags for a specific track (cached per artist+track).

    Returns None on network error so callers can leave the track as NULL for retry.
    Returns [] when the track genuinely has no tags on Last.fm.
    """
    cache_key = (artist, track_name)
    with _lastfm_lock:
        if cache_key in _lastfm_cache:
            return _lastfm_cache[cache_key]
    try:
        data = _lastfm_get({
            "method": "track.getTopTags",
            "artist": artist,
            "track": track_name,
            "api_key": api_key,
            "format": "json",
        })
        if "error" in data:
            if data["error"] == 6:  # track not found — definitive empty result
                tags = []
            else:
                logger.warning("Last.fm track tags API error %d for '%s'/'%s': %s", data["error"], artist, track_name, data.get("message"))
                return None
        else:
            tags = _filter_lastfm_tags(data.get("toptags", {}).get("tag", []))
    except Exception as exc:
        logger.warning("Last.fm track tags failed ('%s' / '%s'): %s", artist, track_name, exc)
        return None  # network error — don't cache; track stays NULL for retry
    with _lastfm_lock:
        _lastfm_cache[cache_key] = tags
    return tags


def _fetch_lastfm_artist_tags(artist: str, api_key: str) -> list[str] | None:
    """Return filtered Last.fm top tags for an artist (cached per artist).

    Returns None on network error so callers can leave the track as NULL for retry.
    Returns [] when the artist genuinely has no tags on Last.fm.
    """
    cache_key = (artist,)
    with _lastfm_lock:
        if cache_key in _lastfm_cache:
            return _lastfm_cache[cache_key]
    try:
        data = _lastfm_get({
            "method": "artist.getTopTags",
            "artist": artist,
            "api_key": api_key,
            "format": "json",
        })
        if "error" in data:
            if data["error"] == 6:  # artist not found — definitive empty result
                tags = []
            else:
                logger.warning("Last.fm artist tags API error %d for '%s': %s", data["error"], artist, data.get("message"))
                return None
        else:
            tags = _filter_lastfm_tags(data.get("toptags", {}).get("tag", []))
    except Exception as exc:
        logger.warning("Last.fm artist tags failed ('%s'): %s", artist, exc)
        return None  # network error — don't cache; track stays NULL for retry
    with _lastfm_lock:
        _lastfm_cache[cache_key] = tags
    return tags


def fetch_lastfm_tags(
    session: Session,
    user: User,
    api_key: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Fetch Last.fm tags for all tracks in the user's library missing them.

    Two-phase parallel approach (_LASTFM_WORKERS threads, shared 5 req/s rate limiter):
      Phase 1 — artist.getTopTags for every unique primary artist, warming the cache.
      Phase 2 — track.getTopTags for every track; artist cache makes fallbacks free.
    Apply order: track tags > artist tags (cached) > genres column > "".

    On network error a helper returns None; that track stays NULL so the next sync
    retries it. progress_callback(done, total) fires after each phase-2 result is
    collected. Returns count of tracks updated.
    """
    tracks = (
        session.query(Track)
        .filter_by(user_id=user.id)
        .filter(Track.lastfm_tags.is_(None) | (Track.lastfm_tags == ""))
        .all()
    )
    total = len(tracks)
    if total == 0:
        logger.info("  Last.fm: no tracks need tagging (all already tagged)")
        return 0

    unique_artists = list({(t.artist or "").split(",")[0].strip() for t in tracks})
    logger.info(f"  Last.fm: {total} tracks, {len(unique_artists)} unique artists ({_LASTFM_WORKERS} workers, ≤5 req/s)")
    t0 = time.monotonic()

    # Phase 1: warm artist cache in parallel — makes phase-2 fallbacks free
    _ARTIST_INTERVAL = max(1, len(unique_artists) // 10)
    with ThreadPoolExecutor(max_workers=_LASTFM_WORKERS) as pool:
        futures = {pool.submit(_fetch_lastfm_artist_tags, a, api_key): a for a in unique_artists}
        done_artists = 0
        for fut in as_completed(futures):
            fut.result()  # result is cached inside _fetch_lastfm_artist_tags
            done_artists += 1
            if done_artists % _ARTIST_INTERVAL == 0 or done_artists == len(unique_artists):
                elapsed = time.monotonic() - t0
                logger.info(f"  Last.fm artist cache: {done_artists}/{len(unique_artists)} in {int(elapsed)}s")

    # Phase 2: per-track lookups for all tracks — prefer over artist tags
    track_tags: dict[tuple[str, str], list[str] | None] = {}
    _TRACK_INTERVAL = max(1, total // 20)
    t1 = time.monotonic()
    logger.info(f"  Last.fm phase 2: per-track lookups for {total} tracks")
    with ThreadPoolExecutor(max_workers=_LASTFM_WORKERS) as pool:
        futures2 = {
            pool.submit(
                _fetch_lastfm_track_tags,
                (t.artist or "").split(",")[0].strip(),
                t.name or "",
                api_key,
            ): t
            for t in tracks
        }
        done_tracks = 0
        for fut in as_completed(futures2):
            track = futures2[fut]
            pa = (track.artist or "").split(",")[0].strip()
            track_tags[(pa, track.name or "")] = fut.result()
            done_tracks += 1
            if done_tracks % _TRACK_INTERVAL == 0 or done_tracks == total:
                elapsed = time.monotonic() - t1
                logger.info(f"  Last.fm tracks: {done_tracks}/{total} in {int(elapsed)}s")
            if progress_callback:
                progress_callback(done_tracks, total)

    # Apply: track tags > artist cache > genres > ""
    count = 0
    _COMMIT_INTERVAL = 100
    for i, track in enumerate(tracks):
        primary = (track.artist or "").split(",")[0].strip()
        track_result = track_tags.get((primary, track.name or ""))
        artist_result = _lastfm_cache.get((primary,))  # free — populated in phase 1

        if track_result is None and artist_result is None:
            continue  # both network errors — leave NULL for retry

        if track_result:
            tags = track_result
        elif artist_result:
            tags = artist_result
        else:
            tags = []

        if not tags and track.genres:
            tags = [g.strip().lower() for g in track.genres.split(",") if g.strip()]

        track.lastfm_tags = ", ".join(tags)
        count += 1

        if (i + 1) % _COMMIT_INTERVAL == 0:
            session.commit()

    session.commit()
    elapsed = time.monotonic() - t0
    logger.info(f"  Last.fm done: {count} tracks tagged in {int(elapsed)}s")
    logger.info("Last.fm tags fetched for %d tracks", count)
    return count


def load_lastfm_username() -> str:
    """Read LASTFM_USERNAME from environment; raise RuntimeError if missing."""
    username = os.environ.get("LASTFM_USERNAME")
    if not username:
        raise RuntimeError("LASTFM_USERNAME is not set. Add it to your .env file.")
    return username


def fetch_scrobble_history(
    session: Session,
    user: User,
    api_key: str,
    username: str,
    months: int = 6,
    refetch: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Page through Last.fm user.getRecentTracks and store scrobbles in scrobble_history.

    Incremental by default: resumes from the most recent stored scrobble so subsequent
    syncs only fetch new plays. Set refetch=True to clear and re-import the full window.

    Scrobbles are matched to library tracks by (primary_artist, track_name) for tag
    join. Unmatched scrobbles are stored with track_id=NULL and still contribute
    temporal signal. Returns count of new rows written.
    """
    from sqlalchemy import func

    if refetch:
        session.query(ScrobbleHistory).filter_by(user_id=user.id).delete()
        session.commit()
        from_dt = datetime.utcnow() - timedelta(days=months * 30)
    else:
        last_ts = session.query(func.max(ScrobbleHistory.played_at)).filter_by(user_id=user.id).scalar()
        if last_ts:
            from_dt = last_ts - timedelta(hours=1)  # 1-hour overlap to catch stragglers
        else:
            from_dt = datetime.utcnow() - timedelta(days=months * 30)

    from_ts = int(from_dt.timestamp())

    # Build library lookup: (primary_artist_lower, track_name_lower) -> track_id
    tracks = session.query(Track).filter_by(user_id=user.id).all()
    track_lookup: dict[tuple[str, str], int] = {}
    for t in tracks:
        primary = (t.artist or "").split(",")[0].strip().lower()
        track_lookup[(primary, (t.name or "").lower())] = t.id

    page = 1
    total_pages = 1
    total_written = 0

    logger.info(f"  Scrobbles: fetching for '{username}' from {from_dt.strftime('%Y-%m-%d')}...")

    while page <= total_pages:
        data = _lastfm_get({
            "method": "user.getRecentTracks",
            "user": username,
            "api_key": api_key,
            "from": from_ts,
            "limit": 200,
            "page": page,
            "format": "json",
        })

        if "error" in data:
            logger.warning("Last.fm scrobble API error %d: %s", data["error"], data.get("message"))
            break

        attr = data["recenttracks"].get("@attr", {})
        if page == 1:
            total_pages = int(attr.get("totalPages", 1) or 1)
            total_scrobbles = int(attr.get("total", 0))
            logger.info(f"  Scrobbles: {total_scrobbles} plays, {total_pages} pages")
            if total_pages == 0:
                break

        items = data["recenttracks"].get("track", [])
        if isinstance(items, dict):  # Last.fm returns a bare dict when there's only one result
            items = [items]
        if not items:
            break

        for item in items:
            if item.get("@attr", {}).get("nowplaying"):
                continue
            uts = (item.get("date") or {}).get("uts")
            if not uts:
                continue
            played_at = datetime.utcfromtimestamp(int(uts))
            artist = ((item.get("artist") or {}).get("#text") or "").strip()
            track_name = (item.get("name") or "").strip()
            if not artist or not track_name:
                continue

            track_id = track_lookup.get((artist.lower(), track_name.lower()))
            stmt = (
                sqlite_insert(ScrobbleHistory)
                .values(
                    user_id=user.id,
                    artist=artist,
                    track_name=track_name,
                    played_at=played_at,
                    track_id=track_id,
                )
                .on_conflict_do_nothing()
            )
            session.execute(stmt)
            total_written += 1

        session.commit()

        if progress_callback:
            progress_callback(page, total_pages)

        page += 1

    logger.info(f"  Scrobbles done: {total_written} new plays stored")
    logger.info("Scrobble history: %d new rows for user %d", total_written, user.id)
    return total_written


def load_scrobble_dataframe(session: Session, user: User) -> pd.DataFrame:
    """
    Load all stored scrobbles for a user, joined to track tags where available.

    Returns a DataFrame with days_ago pre-computed so callers can apply recency
    weights and month-window filtering in Python without re-querying the DB.
    """
    from sqlalchemy.orm import aliased

    rows = (
        session.query(ScrobbleHistory, Track)
        .outerjoin(Track, ScrobbleHistory.track_id == Track.id)
        .filter(ScrobbleHistory.user_id == user.id)
        .order_by(ScrobbleHistory.played_at.desc())
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["played_at", "days_ago", "day_of_week", "hour", "tags", "artist", "track_name"])

    now = datetime.utcnow()
    records = []
    for scrobble, track in rows:
        days_ago = (now - scrobble.played_at).total_seconds() / 86400
        tags = ""
        if track:
            tags = track.lastfm_tags or track.genres or ""
        records.append({
            "played_at": scrobble.played_at,
            "days_ago": days_ago,
            "day_of_week": scrobble.played_at.weekday(),  # 0=Monday
            "hour": scrobble.played_at.hour,
            "tags": tags,
            "artist": scrobble.artist,
            "track_name": scrobble.track_name,
        })

    return pd.DataFrame(records)


def upsert_user(session: Session, sp: spotipy.Spotify) -> User:
    """Fetch the current Spotify profile and upsert into the users table."""
    logger.info("Step 1/5: Fetching Spotify profile...")
    profile = sp.current_user()
    user = session.query(User).filter_by(spotify_id=profile["id"]).first()
    if not user:
        user = User(spotify_id=profile["id"])
        session.add(user)
    user.display_name = profile.get("display_name")
    user.email = profile.get("email") or ""
    session.commit()
    logger.info(f"  Profile: {user.display_name or profile['id']}")
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
    logger.info("Step 2/5: Syncing liked songs...")
    count = 0
    page = 0
    artist_map: dict[str, list[int]] = {}
    results = sp.current_user_saved_tracks(limit=50)
    total_hint = results.get("total") if results else None
    if total_hint:
        logger.info(f"  Liked songs: {total_hint} total on Spotify")
    while results:
        page += 1
        page_count = len(results["items"])
        for item in results["items"]:
            track, artist_id = _upsert_track(session, user.id, item, added_at=item.get("added_at"))
            if artist_id:
                artist_map.setdefault(artist_id, []).append(track.id)
            count += 1
        session.commit()
        logger.info(f"  Page {page}: {page_count} tracks ({count} total so far)")
        if results.get("next"):
            time.sleep(0.1)
            results = sp.next(results)
        else:
            results = None
    logger.info(f"  Liked songs done: {count} tracks saved")
    _enrich_genres_batch(sp, session, artist_map)
    logger.info("Saved tracks synced: %d", count)
    return count


def fetch_top_tracks(sp: spotipy.Spotify, session: Session, user: User) -> int:
    """Fetch top tracks across all three Spotify time ranges and persist them."""
    logger.info("Step 3/5: Syncing top tracks (short / medium / long term)...")
    count = 0
    artist_map: dict[str, list[int]] = {}
    range_labels = {"short_term": "4 weeks", "medium_term": "6 months", "long_term": "all time"}
    for time_range in ("short_term", "medium_term", "long_term"):
        range_count = 0
        page = 0
        logger.info(f"  Time range: {range_labels[time_range]}")
        results = sp.current_user_top_tracks(limit=50, time_range=time_range)
        while results and page < _MAX_TOP_TRACKS_PAGES:
            page += 1
            page_count = len(results["items"])
            for item in results["items"]:
                track, artist_id = _upsert_track(session, user.id, {"track": item})
                if artist_id:
                    artist_map.setdefault(artist_id, []).append(track.id)
                count += 1
                range_count += 1
            session.commit()
            logger.info(f"    Page {page}: {page_count} tracks ({range_count} this range, {count} total)")
            if results.get("next"):
                time.sleep(0.1)
                results = sp.next(results)
            else:
                results = None
        if page >= _MAX_TOP_TRACKS_PAGES and results and results.get("next"):
            logger.info(f"    Stopped at page cap ({_MAX_TOP_TRACKS_PAGES} pages / {range_count} tracks) for {time_range}")
    logger.info(f"  Top tracks done: {count} tracks saved")
    _enrich_genres_batch(sp, session, artist_map)
    logger.info("Top tracks synced: %d", count)
    return count


def fetch_recently_played(sp: spotipy.Spotify, session: Session, user: User) -> int:
    """Fetch up to 50 recent plays (Spotify API limit) with timestamps."""
    logger.info("Step 4/5: Syncing recently played (up to 50)...")
    results = sp.current_user_recently_played(limit=50)
    count = 0
    artist_map: dict[str, list[int]] = {}
    items = results.get("items", [])
    logger.info(f"  Got {len(items)} recent plays from Spotify")
    for item in items:
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
    logger.info(f"  Recent plays done: {count} new history entries")
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
            "cluster": t.cluster_index,
        }
        for t in rows
    ]
    df = pd.DataFrame(records)
    if df["cluster"].isna().all():
        df = df.drop(columns=["cluster"])
    return df


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
