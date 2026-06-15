#!/usr/bin/env python
"""
Tests for fetch.py API call patterns.
No Spotify credentials or network access required — all Spotify calls are mocked.

Key invariant: sp.artist() (singular, the old per-track path) must NEVER be called.
All genre enrichment must go through sp.artists() (batch, max 50 IDs per call).

Run with:  .\\venv\\Scripts\\python tests\\test_fetch.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable sleep globally so tests don't wait on rate-limit spacing
_sleep_patch = patch("app.fetch.time.sleep")
_sleep_patch.start()

import app.fetch as fetch_module
from app.fetch import _enrich_genres_batch


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------

def _sp(artist_genre_map: dict[str, list[str]] | None = None) -> MagicMock:
    """
    Mock Spotipy client.
    sp.artist() raises — it must never be called.
    sp.artists(ids) returns synthetic genre data.
    """
    sp = MagicMock()
    sp.artist.side_effect = AssertionError(
        "sp.artist() (singular) was called — the old per-track path is still active"
    )
    genre_map = artist_genre_map or {}

    def _fake_artists(ids):
        return {
            "artists": [
                {"id": aid, "genres": genre_map.get(aid, ["pop"])}
                for aid in ids
            ]
        }

    sp.artists.side_effect = _fake_artists
    return sp


def _session(track_map: dict[int, MagicMock]) -> MagicMock:
    sess = MagicMock()
    sess.get.side_effect = lambda _model, pk: track_map.get(pk)
    return sess


def _track(track_id: int, genres: str | None = None) -> MagicMock:
    t = MagicMock()
    t.id = track_id
    t.genres = genres
    return t


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def main() -> int:
    passed = 0
    failed = 0

    def ok(label: str) -> None:
        nonlocal passed
        passed += 1
        print(f"  PASS  {label}")

    def fail(label: str, reason: str) -> None:
        nonlocal failed
        failed += 1
        print(f"  FAIL  {label}")
        print(f"        {reason}")

    def run(label: str, fn) -> bool:
        try:
            fn()
            ok(label)
            return True
        except Exception as exc:
            fail(label, str(exc))
            traceback.print_exc()
            return False

    # -- batch call counts ----------------------------------------------------
    print("\n[_enrich_genres_batch - batch call counts]")

    def _test_0_artists():
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(_sp(), _session({}), {})
        # no assertion needed — sp.artist() raising inside _sp() covers it

    run("0 artists -> 0 sp.artists() calls", _test_0_artists)

    def _test_1_artist():
        artist_map = {"a0": [0]}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session({0: _track(0)}), artist_map)
        assert sp.artists.call_count == 1, f"got {sp.artists.call_count}"

    run("1 artist -> 1 sp.artists() call", _test_1_artist)

    def _test_50_artists():
        ids = [f"a{i}" for i in range(50)]
        artist_map = {aid: [i] for i, aid in enumerate(ids)}
        tracks = {i: _track(i) for i in range(50)}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)
        assert sp.artists.call_count == 1, f"got {sp.artists.call_count}"

    run("50 artists -> 1 sp.artists() call (exact boundary)", _test_50_artists)

    def _test_51_artists():
        ids = [f"a{i}" for i in range(51)]
        artist_map = {aid: [i] for i, aid in enumerate(ids)}
        tracks = {i: _track(i) for i in range(51)}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)
        assert sp.artists.call_count == 2, f"got {sp.artists.call_count}"

    run("51 artists -> 2 sp.artists() calls (boundary + 1)", _test_51_artists)

    def _test_300_artists():
        ids = [f"a{i}" for i in range(300)]
        artist_map = {aid: [i] for i, aid in enumerate(ids)}
        tracks = {i: _track(i) for i in range(300)}
        sp = _sp()
        expected = math.ceil(300 / 50)  # 6
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)
        assert sp.artists.call_count == expected, f"expected {expected}, got {sp.artists.call_count}"

    run("300 artists -> 6 sp.artists() calls", _test_300_artists)

    # -- sp.artist() singular never called ------------------------------------
    print("\n[_enrich_genres_batch - sp.artist() singular never called]")

    def _test_singular_never_called_large():
        ids = [f"a{i}" for i in range(120)]
        artist_map = {aid: [i] for i, aid in enumerate(ids)}
        tracks = {i: _track(i) for i in range(120)}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)

    run("120 artists: sp.artist() never called", _test_singular_never_called_large)

    # -- cache prevents re-fetching -------------------------------------------
    print("\n[_enrich_genres_batch - cache hit avoids API calls]")

    def _test_fully_cached():
        cached = {f"a{i}": "rock" for i in range(30)}
        artist_map = {aid: [i] for i, aid in enumerate(cached)}
        tracks = {i: _track(i) for i in range(30)}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, cached, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)
        assert sp.artists.call_count == 0, f"all cached but called {sp.artists.call_count} times"

    run("all 30 artists cached -> 0 API calls", _test_fully_cached)

    def _test_partial_cache():
        cached = {f"a{i}": "pop" for i in range(40)}
        new_ids = [f"new{i}" for i in range(10)]
        artist_map = {
            **{aid: [i] for i, aid in enumerate(cached)},
            **{aid: [40 + i] for i, aid in enumerate(new_ids)},
        }
        tracks = {i: _track(i) for i in range(50)}
        sp = _sp()
        with patch.dict(fetch_module._artist_genre_cache, cached, clear=True):
            _enrich_genres_batch(sp, _session(tracks), artist_map)
        assert sp.artists.call_count == 1, f"expected 1 call for 10 new, got {sp.artists.call_count}"
        called_ids = set(sp.artists.call_args[0][0])
        assert called_ids == set(new_ids), f"wrong IDs batched: {called_ids}"

    run("40 cached + 10 new -> 1 call covering only the 10 new", _test_partial_cache)

    # -- genre data written correctly -----------------------------------------
    print("\n[_enrich_genres_batch - genre data written to tracks]")

    def _test_genres_written():
        genre_map = {
            "artist_rock": ["rock", "classic rock"],
            "artist_pop":  ["pop", "dance pop"],
        }
        track_rock = _track(1)
        track_pop  = _track(2)
        artist_map = {"artist_rock": [1], "artist_pop": [2]}
        sp = _sp(genre_map)
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session({1: track_rock, 2: track_pop}), artist_map)
        assert track_rock.genres == "rock, classic rock", f"got {track_rock.genres!r}"
        assert track_pop.genres  == "pop, dance pop",     f"got {track_pop.genres!r}"

    run("genres written correctly to track objects", _test_genres_written)

    def _test_existing_genres_not_overwritten():
        track0 = _track(0, genres="already set")
        sp = _sp({"a0": ["new genre"]})
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session({0: track0}), {"a0": [0]})
        assert track0.genres == "already set", f"genres were overwritten: {track0.genres!r}"

    run("pre-existing genres not overwritten by batch enrichment", _test_existing_genres_not_overwritten)

    def _test_missing_artist_gets_empty_string():
        # API returns None for the artist (artist deleted / unavailable)
        sp = MagicMock()
        sp.artist.side_effect = AssertionError("singular must not be called")
        sp.artists.return_value = {"artists": [None]}  # Spotify returns null for missing artists
        track0 = _track(0)
        with patch.dict(fetch_module._artist_genre_cache, {}, clear=True):
            _enrich_genres_batch(sp, _session({0: track0}), {"a0": [0]})
        assert track0.genres == "", f"expected empty string for missing artist, got {track0.genres!r}"

    run("missing/deleted artist gets empty-string genres (no crash)", _test_missing_artist_gets_empty_string)

    # -- summary --------------------------------------------------------------
    print(f"\n{'-' * 44}")
    status = "OK" if failed == 0 else "FAILED"
    print(f"  {status}  -  {passed} passed, {failed} failed")
    print("-" * 44)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
