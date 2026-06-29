from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import spotipy
import streamlit as st

from app.auth import exchange_code, get_auth_url, get_spotify_client
from app.cluster import (
    build_feature_matrix,
    genre_distribution,
    infer_current_mood,
    load_cluster_labels,
    pca_projection,
    run_clustering,
)
from app.database import get_session, init_db
from app.daylist import build_week_grid, compute_daylist, get_time_label
from app.fetch import (
    fetch_lastfm_tags,
    fetch_recently_played,
    fetch_saved_tracks,
    fetch_scrobble_history,
    fetch_top_tracks,
    is_genres_forbidden,
    load_history_dataframe,
    load_lastfm_api_key,
    load_lastfm_username,
    load_scrobble_dataframe,
    load_tracks_dataframe,
    reset_genres_forbidden,
    reset_lastfm_tags,
    upsert_user,
)
from app.models import User
from app.recommend import build_playlist, create_or_replace_playlist

logger = logging.getLogger(__name__)

_SESSION_KEYS = ("sp", "user", "spotify_id", "df_clustered", "kmeans", "cluster_labels", "df_scrobbles")


def _init_session() -> None:
    for key in _SESSION_KEYS:
        st.session_state.setdefault(key, None)


def _handle_oauth() -> bool:
    """Exchange an OAuth code when Spotify redirects back. Returns True if a code was handled."""
    code = st.query_params.get("code")
    if not code:
        return False
    try:
        exchange_code(code)
    except Exception as exc:
        st.error(f"OAuth error: {exc}")
        return False
    st.query_params.clear()
    return True


def _try_load_client() -> bool:
    """Load a cached Spotify client into session_state. Returns True on success."""
    if st.session_state.sp is not None:
        return True
    try:
        st.session_state.sp = get_spotify_client()
        return True
    except RuntimeError:
        return False


def _load_app_state(session) -> None:
    """Populate session_state from DB. Only calls Spotify API if no DB user exists yet."""
    spotify_id = st.session_state.spotify_id

    if not spotify_id:
        # Check DB first — avoids an API call after the first sync has run
        existing = session.query(User).first()
        if existing:
            spotify_id = existing.spotify_id
            st.session_state.spotify_id = spotify_id
        else:
            # Brand new user with no sync history: need API to identify them
            try:
                profile = st.session_state.sp.current_user()
                spotify_id = profile["id"]
                st.session_state.spotify_id = spotify_id
            except Exception:
                return

    user = session.query(User).filter_by(spotify_id=spotify_id).first()
    st.session_state.user = user
    if user is None:
        return
    df = load_tracks_dataframe(session, user)
    st.session_state.df_clustered = df
    st.session_state.cluster_labels = load_cluster_labels(session, user)


def render_login() -> None:
    st.title("Spotify Mood Engine")
    st.write(
        "Discover your listening moods and generate playlists that match how you feel right now."
    )
    st.link_button("Connect with Spotify", get_auth_url(), type="primary")


def render_sidebar(session) -> None:
    with st.sidebar:
        st.header("Controls")
        sp = st.session_state.sp
        user = st.session_state.user
        df = st.session_state.df_clustered
        has_tracks = df is not None and not df.empty

        try:
            lastfm_api_key = load_lastfm_api_key()
        except RuntimeError:
            lastfm_api_key = None
            st.warning("Add LASTFM_API_KEY to .env to enable mood tag enrichment")

        refetch_tags = lastfm_api_key and st.checkbox(
            "Re-fetch all Last.fm tags",
            help="Clears cached tags and re-fetches with current settings. Use after changing your API key or to pick up richer tags.",
        )

        if st.button("Sync Library", use_container_width=True):
            reset_genres_forbidden()
            progress = st.progress(0, text="Connecting to Spotify…")
            try:
                user = upsert_user(session, sp)
                if refetch_tags:
                    reset_lastfm_tags(session, user)
                progress.progress(15, text="Syncing liked songs…")
                fetch_saved_tracks(sp, session, user)
                progress.progress(40, text="Syncing top tracks…")
                fetch_top_tracks(sp, session, user)
                progress.progress(65, text="Syncing recent plays…")
                fetch_recently_played(sp, session, user)
                if lastfm_api_key:
                    logger.info("Step 5/5: Fetching Last.fm mood tags...")
                    progress.progress(80, text="Fetching Last.fm tags…")
                    _t0 = time.monotonic()

                    def _lastfm_cb(done: int, total: int) -> None:
                        elapsed = time.monotonic() - _t0
                        rate = done / elapsed if elapsed > 0 else 0
                        remaining = (total - done) / rate if rate > 0 else None
                        pct = 80 + int(done * 20 / total)
                        if remaining is not None:
                            m, s = divmod(int(remaining), 60)
                            eta = f"{m}m {s}s" if m else f"{s}s"
                            label = f"Last.fm tags: {done}/{total} · ETA {eta}"
                        else:
                            label = f"Last.fm tags: {done}/{total}"
                        progress.progress(pct, text=label)

                    fetch_lastfm_tags(session, user, lastfm_api_key, progress_callback=_lastfm_cb)
                progress.progress(100, text="Done!")
                st.session_state.user = user
                df = load_tracks_dataframe(session, user)
                st.session_state.df_clustered = df
                st.session_state.cluster_labels = load_cluster_labels(session, user)

                if is_genres_forbidden():
                    st.warning(
                        "Spotify blocked artist genre lookup (403) — genres were not collected. "
                        "Clustering will rely on Last.fm tags only."
                        + ("" if lastfm_api_key else " Add LASTFM_API_KEY to .env and sync again.")
                    )

                has_any_tags = (
                    df["genres"].str.strip().ne("") | df["lastfm_tags"].str.strip().ne("")
                ).any() if not df.empty else False
                if not has_any_tags:
                    st.error(
                        "No tag data was collected — Re-cluster won't work yet. "
                        + ("Check the console for Last.fm errors." if lastfm_api_key else "Add LASTFM_API_KEY to .env and sync again.")
                    )
                else:
                    st.success(f"Synced {len(df)} tracks.")
                st.rerun()
            except spotipy.exceptions.SpotifyException as exc:
                if exc.http_status == 429:
                    retry_after = getattr(exc, "headers", {}) or {}
                    retry_secs = retry_after.get("Retry-After") or retry_after.get("retry-after")
                    if retry_secs:
                        try:
                            secs = int(retry_secs)
                            h, m = divmod(secs, 3600)
                            m //= 60
                            wait_str = f"{h}h {m}m" if h else f"{m}m"
                            logger.warning("RATE LIMITED — retry after %ds (~%s)", secs, wait_str)
                            st.error(f"Spotify rate limit — blocked for ~{wait_str}. Try again later.")
                        except (ValueError, TypeError):
                            st.error("Spotify rate limit reached. Wait before trying to sync again.")
                    else:
                        logger.warning("RATE LIMITED (no Retry-After header)")
                        st.error("Spotify rate limit reached. Wait before trying to sync again.")
                else:
                    st.error(f"Spotify error {exc.http_status}: {exc.msg}")
            except Exception as exc:
                logger.exception("Sync failed")
                st.error(f"Sync failed: {exc}")
        st.caption("Sync fetches Spotify library + Last.fm mood tags")

        st.divider()
        st.subheader("Scrobble History")

        try:
            lastfm_username = load_lastfm_username()
        except RuntimeError:
            lastfm_username = None
            st.warning("Add LASTFM_USERNAME to .env to enable scrobble history")

        scrobble_months = st.slider(
            "History window (months)",
            min_value=1, max_value=24, value=6,
            key="scrobble_months_slider",
            help="How far back to import scrobbles. Longer windows capture seasonal patterns but may dilute recent taste shifts.",
        )
        half_life_days = st.slider(
            "Recency half-life (days)",
            min_value=7, max_value=90, value=30,
            key="half_life_slider",
            help="Plays this many days ago count half as much as today's plays. Lower = more reactive to recent habits.",
        )

        refetch_scrobbles = st.checkbox(
            "Re-import full window",
            help="Clear existing scrobbles and re-import the entire history window. Use when changing the window size.",
        )

        scrobble_count = 0
        if st.session_state.df_scrobbles is not None:
            scrobble_count = len(st.session_state.df_scrobbles)

        if lastfm_username and st.button("Sync Scrobble History", use_container_width=True):
            lastfm_api_key_for_scrobbles = load_lastfm_api_key() if lastfm_api_key else None
            if not lastfm_api_key_for_scrobbles:
                st.error("LASTFM_API_KEY required to sync scrobbles.")
            else:
                scrobble_progress = st.progress(0, text="Connecting to Last.fm…")

                def _scrobble_cb(page: int, total: int) -> None:
                    pct = int(page * 100 / max(total, 1))
                    scrobble_progress.progress(pct, text=f"Importing scrobbles: page {page}/{total}")

                try:
                    user = st.session_state.user
                    if user is None:
                        st.error("Sync your Spotify library first.")
                    else:
                        n_new = fetch_scrobble_history(
                            session, user, lastfm_api_key_for_scrobbles,
                            lastfm_username, scrobble_months, refetch_scrobbles, _scrobble_cb,
                        )
                        scrobble_progress.progress(100, text="Done!")
                        st.session_state.df_scrobbles = load_scrobble_dataframe(session, user)
                        scrobble_count = len(st.session_state.df_scrobbles)
                        st.success(f"{n_new} new plays imported ({scrobble_count} total).")
                        st.rerun()
                except Exception as exc:
                    logger.exception("Scrobble sync failed")
                    st.error(f"Scrobble sync failed: {exc}")

        if scrobble_count:
            st.caption(f"{scrobble_count:,} scrobbles stored")

        st.divider()

        auto_k = st.checkbox(
            "Auto-detect best K",
            value=True,
            help="Sweep K from 2–10 and pick the value with the highest silhouette score. Uncheck to set K manually.",
        )
        n_clusters = st.slider(
            "Mood clusters",
            min_value=2,
            max_value=10,
            value=5,
            disabled=auto_k,
            help="Number of mood clusters. Only used when Auto-detect is off.",
        )

        if st.button(
            "Re-cluster",
            disabled=not has_tracks,
            use_container_width=True,
            help="Run K-means to find mood clusters in your library.",
        ):
            has_tags = (
                df["genres"].str.strip().ne("") | df["lastfm_tags"].str.strip().ne("")
            ).any()
            if not has_tags:
                st.warning(
                    "No mood tags yet — finish syncing Last.fm tags before clustering."
                )
            else:
                with st.spinner("Clustering…"):
                    k_min, k_max = (2, 10) if auto_k else (n_clusters, n_clusters)
                    df_c, kmeans, _scaler, scores = run_clustering(
                        df, session, user, k_min=k_min, k_max=k_max
                    )
                    st.session_state.df_clustered = df_c
                    st.session_state.kmeans = kmeans
                    st.session_state.cluster_labels = load_cluster_labels(session, user)
                best_k = len(st.session_state.cluster_labels)
                best_sil = scores.get(best_k, 0.0) if scores else 0.0
                st.success(f"Found {best_k} mood clusters (silhouette: {best_sil:.3f}).")
                st.rerun()

        st.divider()

        if user:
            st.write(f"**{user.display_name or user.spotify_id}**")

        labels = st.session_state.cluster_labels
        df_c = st.session_state.df_clustered
        if (
            labels
            and df_c is not None
            and not df_c.empty
            and "cluster" in df_c.columns
        ):
            df_hist = load_history_dataframe(session, user)
            mood_idx = infer_current_mood(df_c, df_hist, datetime.now().hour)
            st.metric("Current Mood", labels.get(mood_idx, f"Cluster {mood_idx}"))


def render_daylist(session) -> None:
    user = st.session_state.user
    if user is None:
        return

    # Lazy-load scrobbles if not yet in session state
    if st.session_state.df_scrobbles is None:
        df_raw = load_scrobble_dataframe(session, user)
        if not df_raw.empty:
            st.session_state.df_scrobbles = df_raw

    df = st.session_state.df_scrobbles
    if df is None or df.empty:
        with st.expander("Your Daylist  *(sync scrobble history to enable)*", expanded=False):
            st.info("No scrobble data yet — use **Sync Scrobble History** in the sidebar.")
        return

    st.subheader("Your Daylist")

    now = datetime.now()

    # Controls in a row
    col_day, col_hour, col_opts = st.columns([2, 2, 1])
    with col_day:
        day_options = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        selected_dow = st.selectbox(
            "Day", day_options, index=now.weekday(),
            key="daylist_dow",
        )
        target_dow = day_options.index(selected_dow)
    with col_hour:
        target_hour = st.slider("Hour", 0, 23, now.hour, key="daylist_hour")
    months = st.session_state.get("scrobble_months_slider", 6)
    half_life = st.session_state.get("half_life_slider", 30)

    result = compute_daylist(df, months, half_life, target_hour, target_dow)

    # Prominent descriptor
    st.markdown(f"### {result['descriptor']}")
    if result.get("fallback_level", 0) == 1:
        st.caption(f"_Not enough plays on {selected_dow}s at this hour — showing all {result['time_label']}s across every day._")
    elif result.get("fallback_level", 0) == 2:
        st.caption("_No plays found for this time window — showing your full listening history._")

    meta_col, tags_col = st.columns([1, 2])
    with meta_col:
        time_label = get_time_label(target_hour)
        st.caption(
            f"**{result['n_plays']:,}** plays in this window  ·  "
            f"**{result['n_total']:,}** total in {months}-month window"
        )
        if result["vibe"]:
            st.caption(f"Vibe: **{result['vibe']}**")
        if result["genre"]:
            st.caption(f"Genre: **{result['genre']}**")

    with tags_col:
        if result["top_tags"]:
            scores = result["top_scores"]
            total_weight = sum(scores.values()) or 1
            tag_str = "  ·  ".join(
                f"**{t}** ({scores[t]/total_weight*100:.0f}%)" for t in result["top_tags"][:7]
            )
            st.caption(f"Top tags: {tag_str}")

    # Contributing tracks table
    subset = result.get("subset")
    if subset is not None and not subset.empty:
        display = subset[["track_name", "artist", "tags", "played_at", "weight"]].copy()
        display["tags"] = display["tags"].apply(
            lambda s: ", ".join(t.strip() for t in (s or "").split(",")[:4] if t.strip()) or "—"
        )
        display["played_at"] = pd.to_datetime(display["played_at"]).dt.strftime("%Y-%m-%d %H:%M")
        display["weight"] = display["weight"].round(3)
        display = display.rename(columns={
            "track_name": "Track",
            "artist": "Artist",
            "tags": "Genre / Tags",
            "played_at": "Listened",
            "weight": "Weight",
        })
        with st.expander(f"Contributing tracks ({len(display):,})", expanded=True):
            st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("Full week grid — all hours · 6 months · 30d half-life", expanded=False):
        _day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        grid = build_week_grid(df, 6, 30)

        st.caption("Descriptor (day × hour)")
        desc_pivot = grid.pivot(index="day", columns="hour", values="descriptor").reindex(_day_order)
        st.dataframe(desc_pivot, use_container_width=True)

        st.caption("Play count (day × hour)")
        plays_pivot = grid.pivot(index="day", columns="hour", values="n_plays").reindex(_day_order)
        st.dataframe(plays_pivot, use_container_width=True)

        st.caption("Fallback level (0 = specific day+hour, 1 = time-slot across all days, 2 = all data)")
        fb_pivot = grid.pivot(index="day", columns="hour", values="fallback").reindex(_day_order)
        st.dataframe(fb_pivot, use_container_width=True)

    st.divider()


def render_charts(session) -> None:
    df = st.session_state.df_clustered
    labels = st.session_state.cluster_labels or {}

    if df is None or df.empty or "cluster" not in df.columns:
        st.info(
            "No mood clusters yet — sync your library then click **Re-cluster** in the sidebar."
        )
        return

    X, *_ = build_feature_matrix(df)
    df_pca = pca_projection(df, X)
    df_pca["mood"] = df_pca["cluster"].map(labels).fillna(df_pca["cluster"].astype(str))

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Mood Clusters (PCA)")
        fig = px.scatter(
            df_pca,
            x="pc1",
            y="pc2",
            color="mood",
            hover_data=["name", "artist"],
            labels={"pc1": "PC 1", "pc2": "PC 2", "mood": "Mood"},
            height=420,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.75))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Genres per Cluster")
        genre_df = genre_distribution(df, top_n=5)
        if not genre_df.empty:
            genre_df["mood"] = genre_df["cluster_index"].map(
                lambda i: labels.get(i, f"Cluster {i}")
            )
            fig_g = px.bar(
                genre_df,
                x="count",
                y="genre",
                color="mood",
                orientation="h",
                barmode="group",
                labels={"count": "Tracks", "genre": "Genre", "mood": "Mood"},
                height=420,
            )
            fig_g.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("No genre data available — sync your library first.")


def render_playlist_section(session) -> None:
    labels = st.session_state.cluster_labels or {}
    df = st.session_state.df_clustered
    sp = st.session_state.sp
    user = st.session_state.user

    if not labels or df is None or df.empty or "cluster" not in df.columns:
        return

    st.subheader("Generate Playlist")

    # Deduplicate mood labels: append cluster index when two clusters share a name.
    display_options: dict[str, int] = {}
    seen: dict[str, int] = {}
    for idx, lbl in sorted(labels.items()):
        if lbl in seen:
            orig_idx = seen[lbl]
            if lbl in display_options:
                display_options.pop(lbl)
                display_options[f"{lbl} ({orig_idx})"] = orig_idx
            display_options[f"{lbl} ({idx})"] = idx
        else:
            display_options[lbl] = idx
            seen[lbl] = idx

    col_sel, col_size = st.columns([2, 1])
    with col_sel:
        selected_label = st.selectbox("Mood", list(display_options.keys()))
    with col_size:
        target_size = st.number_input("Tracks", min_value=5, max_value=100, value=30)

    if st.button("Create Playlist on Spotify", type="primary"):
        cluster_idx = display_options[selected_label]
        base_label = labels[cluster_idx]
        with st.spinner("Building playlist…"):
            track_ids = build_playlist(df, cluster_idx, int(target_size))
        if not track_ids:
            st.warning("No tracks found for this mood cluster.")
            return
        with st.spinner("Saving playlist to Spotify…"):
            url = create_or_replace_playlist(
                sp, user.spotify_id, track_ids, base_label
            )
        st.success(
            f"Playlist ready — **{len(track_ids)} tracks** added. "
            f"[Open in Spotify]({url})"
        )


def render_tag_explorer() -> None:
    df = st.session_state.df_clustered
    labels = st.session_state.cluster_labels or {}

    if df is None or df.empty:
        return

    with st.expander("Tag / Genre Explorer"):
        # Build tag -> track_id index in one pass
        tag_counts: Counter = Counter()
        tag_to_ids: dict[str, list[int]] = {}
        for _, row in df.iterrows():
            tid = int(row["track_id"])
            for col in ("genres", "lastfm_tags"):
                for raw in (row.get(col) or "").split(","):
                    tag = raw.strip().lower()
                    if tag:
                        tag_counts[tag] += 1
                        tag_to_ids.setdefault(tag, []).append(tid)

        if not tag_counts:
            st.info("No tags found — sync your library to populate genre and Last.fm data.")
            return

        sorted_tags = [tag for tag, _ in tag_counts.most_common()]

        selected_tag = st.selectbox(
            "Browse by tag",
            sorted_tags,
            format_func=lambda t: f"{t}  ({tag_counts[t]} tracks)",
        )

        if selected_tag:
            ids = set(tag_to_ids[selected_tag])
            matching = df[df["track_id"].isin(ids)].copy()

            st.caption(f"{len(matching)} tracks tagged **{selected_tag}**")

            if "cluster" in matching.columns:
                matching["Mood"] = matching["cluster"].map(labels).fillna("—")
                display = matching[["name", "artist", "Mood"]].rename(
                    columns={"name": "Track", "artist": "Artist"}
                )
            else:
                display = matching[["name", "artist"]].rename(
                    columns={"name": "Track", "artist": "Artist"}
                )

            st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Spotify Mood Engine",
        page_icon=":musical_note:",
        layout="wide",
    )
    init_db()
    _init_session()

    if _handle_oauth():
        st.rerun()

    if not _try_load_client():
        render_login()
        st.stop()

    session = get_session()
    try:
        if st.session_state.user is None:
            _load_app_state(session)

        st.title("Spotify Mood Engine")
        render_sidebar(session)
        render_daylist(session)
        render_charts(session)
        render_tag_explorer()
        render_playlist_section(session)
    finally:
        session.close()
