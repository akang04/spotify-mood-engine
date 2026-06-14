#!/usr/bin/env python
"""
Smoke test for the clustering pipeline.
Uses a synthetic dataset - no Spotify credentials, no Last.fm API, no database.

Usage:
    .\\venv\\Scripts\\python tests\\smoke.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# fmt: off
# Three clear clusters (hip-hop / chill / rock) plus edge-case tracks.
# Tags are split across genres and lastfm_tags to exercise the merge path.
_TRACKS = [
    # hip-hop
    {"track_id": 1,  "spotify_track_id": "s1",  "name": "Track 01", "artist": "A", "album": "X", "popularity": 80, "genres": "hip hop, rap",       "lastfm_tags": "hip hop, rap, trap"},
    {"track_id": 2,  "spotify_track_id": "s2",  "name": "Track 02", "artist": "B", "album": "X", "popularity": 75, "genres": "hip hop",             "lastfm_tags": "rap, hip hop"},
    {"track_id": 3,  "spotify_track_id": "s3",  "name": "Track 03", "artist": "C", "album": "X", "popularity": 70, "genres": "trap, hip hop",       "lastfm_tags": "trap"},
    {"track_id": 4,  "spotify_track_id": "s4",  "name": "Track 04", "artist": "D", "album": "X", "popularity": 65, "genres": "rap",                 "lastfm_tags": "hip hop, rap"},
    {"track_id": 5,  "spotify_track_id": "s5",  "name": "Track 05", "artist": "E", "album": "X", "popularity": 72, "genres": "hip hop, drill",      "lastfm_tags": "drill, hip hop"},
    # chill / lo-fi
    {"track_id": 6,  "spotify_track_id": "s6",  "name": "Track 06", "artist": "F", "album": "Y", "popularity": 60, "genres": "lo-fi",               "lastfm_tags": "chill, study, lo-fi"},
    {"track_id": 7,  "spotify_track_id": "s7",  "name": "Track 07", "artist": "G", "album": "Y", "popularity": 55, "genres": "ambient",             "lastfm_tags": "ambient, chill, relaxing"},
    {"track_id": 8,  "spotify_track_id": "s8",  "name": "Track 08", "artist": "H", "album": "Y", "popularity": 58, "genres": "lo-fi, chillhop",     "lastfm_tags": "chillhop, study"},
    {"track_id": 9,  "spotify_track_id": "s9",  "name": "Track 09", "artist": "I", "album": "Y", "popularity": 50, "genres": "ambient",             "lastfm_tags": "peaceful, chill"},
    {"track_id": 10, "spotify_track_id": "s10", "name": "Track 10", "artist": "J", "album": "Y", "popularity": 52, "genres": "lo-fi",               "lastfm_tags": "focus, study, chill"},
    # rock
    {"track_id": 11, "spotify_track_id": "s11", "name": "Track 11", "artist": "K", "album": "Z", "popularity": 78, "genres": "rock",                "lastfm_tags": "rock, classic rock"},
    {"track_id": 12, "spotify_track_id": "s12", "name": "Track 12", "artist": "L", "album": "Z", "popularity": 82, "genres": "rock, indie rock",    "lastfm_tags": "indie rock, rock"},
    {"track_id": 13, "spotify_track_id": "s13", "name": "Track 13", "artist": "M", "album": "Z", "popularity": 76, "genres": "alternative, rock",   "lastfm_tags": "alternative, rock"},
    {"track_id": 14, "spotify_track_id": "s14", "name": "Track 14", "artist": "N", "album": "Z", "popularity": 80, "genres": "hard rock",           "lastfm_tags": "rock, hard rock"},
    {"track_id": 15, "spotify_track_id": "s15", "name": "Track 15", "artist": "O", "album": "Z", "popularity": 74, "genres": "rock",                "lastfm_tags": "rock, energetic"},
    # edge cases
    {"track_id": 16, "spotify_track_id": "s16", "name": "Track 16", "artist": "P", "album": "W", "popularity":  0, "genres": "",                    "lastfm_tags": ""},                  # no tags ->imputed
    {"track_id": 17, "spotify_track_id": "s17", "name": "Track 17", "artist": "Q", "album": "W", "popularity":  0, "genres": "pop",                 "lastfm_tags": ""},                  # genres only, no lastfm
    {"track_id": 18, "spotify_track_id": "s18", "name": "Track 18", "artist": "R", "album": "W", "popularity":  0, "genres": "",                    "lastfm_tags": "happy, feel good"},  # lastfm only, no genres
]
# fmt: on


def main() -> int:
    from app.cluster import (
        assign_clusters,
        build_feature_matrix,
        genre_distribution,
        infer_mood_label,
        pca_projection,
        run_kmeans,
        select_k,
    )

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

    df = pd.DataFrame(_TRACKS)
    X = scaler = top_tags = None
    best_k = scores = None
    df_clustered = None

    # -- build_feature_matrix -------------------------------------------------
    print("\n[build_feature_matrix]")

    def _build():
        nonlocal X, scaler, top_tags
        X, scaler, top_tags = build_feature_matrix(df)

    if not run("returns without error", _build):
        print("\nCritical failure in build_feature_matrix - aborting.\n")
        return 1

    run("row count matches track count",
        lambda: _assert(X.shape[0] == len(df), f"{X.shape[0]} != {len(df)}"))
    run("top_tags includes 'hip hop'",      lambda: _assert_in("hip hop", top_tags))
    run("top_tags includes 'rock'",         lambda: _assert_in("rock", top_tags))
    run("top_tags includes 'chill'",        lambda: _assert_in("chill", top_tags))
    run("no-tag track (row 15) imputed nonzero",
        lambda: _assert(
            any(v > 0 for v in scaler.inverse_transform(X[15:16])[0]),
            "row 15 raw values all zero after imputation"
        ))
    run("lastfm-only track (row 17) has signal",
        lambda: _assert(X[17].sum() > 0, f"row 17 sum={X[17].sum():.3f}"))

    # -- select_k -------------------------------------------------------------
    print("\n[select_k]")

    def _select():
        nonlocal best_k, scores
        best_k, scores = select_k(X)

    run("runs without error", _select)
    run("best_k in valid range [2, 10]",
        lambda: _assert(best_k is not None and 2 <= best_k <= 10, f"best_k={best_k}"))
    run("scores dict populated",
        lambda: _assert(bool(scores), "scores is empty"))

    # -- run_kmeans + assign_clusters -----------------------------------------
    print("\n[run_kmeans / assign_clusters]")

    def _cluster():
        nonlocal df_clustered
        k = best_k if best_k is not None else 3
        km = run_kmeans(X, k)
        df_clustered = assign_clusters(df, km.labels_)

    run("pipeline completes without error", _cluster)
    run("'cluster' column present",
        lambda: _assert(df_clustered is not None and "cluster" in df_clustered.columns,
                        "missing 'cluster' column"))
    run("all labels in [0, k-1]",
        lambda: _assert(
            df_clustered["cluster"].between(0, (best_k or 3) - 1).all(),
            f"out-of-range label found"))

    # -- genre_distribution ---------------------------------------------------
    print("\n[genre_distribution]")

    def _genre_dist():
        _assert(df_clustered is not None, "df_clustered not available")
        gdf = genre_distribution(df_clustered)
        _assert(not gdf.empty, "returned empty DataFrame")
        _assert({"cluster_index", "genre", "count"} <= set(gdf.columns),
                f"unexpected columns: {gdf.columns.tolist()}")
        # Regression check: genre_distribution must use the merged tag set,
        # not just the genres column. These tags only exist in lastfm_tags.
        all_tags = set(gdf["genre"])
        lastfm_only_tags = {"chill", "chillhop", "rap", "trap", "study", "relaxing"}
        _assert(any(t in all_tags for t in lastfm_only_tags),
                f"no Last.fm tags found - merged tag set not used. Got: {sorted(all_tags)[:10]}")

    run("uses merged genres + lastfm_tags (regression check)", _genre_dist)

    # -- infer_mood_label - rule ordering -------------------------------------
    print("\n[infer_mood_label - rule ordering]")
    run("hip hop ->Hip-Hop",
        lambda: _assert_eq(infer_mood_label(["hip hop"]), "Hip-Hop"))
    run("rock ->Rock",
        lambda: _assert_eq(infer_mood_label(["rock"]), "Rock"))
    run("chill ->Chill",
        lambda: _assert_eq(infer_mood_label(["chill"]), "Chill"))
    run("chillhop ->Chill / Lo-Fi  (specific before broad 'chill')",
        lambda: _assert_eq(infer_mood_label(["chillhop"]), "Chill / Lo-Fi"))
    run("chillout ->Chill / Lo-Fi  (specific before broad 'chill')",
        lambda: _assert_eq(infer_mood_label(["chillout"]), "Chill / Lo-Fi"))
    run("dance pop ->Dance / Pop  (specific before broad 'dance')",
        lambda: _assert_eq(infer_mood_label(["dance pop"]), "Dance / Pop"))
    run("dance ->Dance / Party",
        lambda: _assert_eq(infer_mood_label(["dance"]), "Dance / Party"))
    run("ambient pop ->Chill",
        lambda: _assert_eq(infer_mood_label(["ambient pop"]), "Chill"))
    run("unknown tag ->Mixed",
        lambda: _assert_eq(infer_mood_label(["xyzzy"]), "Mixed"))
    run("empty list ->Mixed",
        lambda: _assert_eq(infer_mood_label([]), "Mixed"))

    # -- pca_projection -------------------------------------------------------
    print("\n[pca_projection]")

    def _pca():
        _assert(df_clustered is not None, "df_clustered not available")
        out = pca_projection(df_clustered, X)
        _assert(out.shape == (len(df), 5), f"expected ({len(df)}, 5), got {out.shape}")
        _assert({"pc1", "pc2"} <= set(out.columns), f"missing pca columns: {out.columns.tolist()}")

    run("returns correct shape with pc1/pc2 columns", _pca)

    # -- summary --------------------------------------------------------------
    print(f"\n{'-' * 44}")
    status = "OK" if failed == 0 else "FAILED"
    print(f"  {status}  -  {passed} passed, {failed} failed")
    print("-" * 44)
    return 0 if failed == 0 else 1


def _assert(cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg)


def _assert_in(item: object, collection: object) -> None:
    if item not in collection:
        raise AssertionError(f"{item!r} not found in collection")


def _assert_eq(actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    sys.exit(main())
