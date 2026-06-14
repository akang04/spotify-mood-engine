from __future__ import annotations

import logging
from datetime import datetime

import plotly.express as px
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
from app.fetch import (
    fetch_lastfm_tags,
    fetch_recently_played,
    fetch_saved_tracks,
    fetch_top_tracks,
    load_history_dataframe,
    load_lastfm_api_key,
    load_tracks_dataframe,
    upsert_user,
)
from app.models import User
from app.recommend import build_playlist, create_or_replace_playlist

logger = logging.getLogger(__name__)

_SESSION_KEYS = ("sp", "user", "spotify_id", "df_clustered", "kmeans", "scaler", "cluster_labels")


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

        if st.button("Sync Library", use_container_width=True):
            progress = st.progress(0, text="Connecting to Spotify…")
            user = upsert_user(session, sp)
            progress.progress(15, text="Syncing liked songs…")
            fetch_saved_tracks(sp, session, user)
            progress.progress(40, text="Syncing top tracks…")
            fetch_top_tracks(sp, session, user)
            progress.progress(65, text="Syncing recent plays…")
            fetch_recently_played(sp, session, user)
            if lastfm_api_key:
                progress.progress(80, text="Fetching Last.fm tags… (this may take a bit for large libraries)")
                fetch_lastfm_tags(session, user, lastfm_api_key)
            progress.progress(100, text="Done!")
            st.session_state.user = user
            df = load_tracks_dataframe(session, user)
            st.session_state.df_clustered = df
            st.session_state.cluster_labels = load_cluster_labels(session, user)
            st.success(f"Synced {len(df)} tracks.")
            st.rerun()
        st.caption("Sync fetches Spotify library + Last.fm mood tags")

        if st.button(
            "Re-cluster",
            disabled=not has_tracks,
            use_container_width=True,
            help="Run K-means to find mood clusters in your library.",
        ):
            with st.spinner("Clustering…"):
                df_c, kmeans, scaler, _ = run_clustering(df, session, user)
                st.session_state.df_clustered = df_c
                st.session_state.kmeans = kmeans
                st.session_state.scaler = scaler
                st.session_state.cluster_labels = load_cluster_labels(session, user)
            st.success(f"Found {len(st.session_state.cluster_labels)} mood clusters.")
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
        render_charts(session)
        render_playlist_section(session)
    finally:
        session.close()
