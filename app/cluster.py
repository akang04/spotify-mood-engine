from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.models import Cluster, Track, User

logger = logging.getLogger(__name__)

GENRE_TOP_N = 50
K_MIN = 2
K_MAX = 10

# Ordered rules: first substring match in the cluster's top-tag list wins.
# Mood-adjacent Last.fm tags come first (more specific than genre names);
# within the genre section, multi-word rules appear BEFORE their component words
# (e.g. "indie pop" before "pop", "indie rock" before "rock") to avoid false matches.
_GENRE_MOOD_RULES: list[tuple[str, str]] = [
    # Mood-adjacent Last.fm tags — checked before genre rules
    ("melancholic",         "Melancholic"),
    ("rainy day",           "Melancholic"),
    ("sad",                 "Melancholic"),
    ("heartbreak",          "Melancholic"),
    ("dark",                "Melancholic"),
    ("chillhop",            "Chill / Lo-Fi"),
    ("chillout",            "Chill / Lo-Fi"),
    ("chill",               "Chill"),
    ("relaxing",            "Chill"),
    ("calm",                "Chill"),
    ("peaceful",            "Chill"),
    ("ambient pop",         "Chill"),
    ("ambient",             "Chill"),
    ("driving",             "Energetic"),
    ("workout",             "Energetic"),
    ("energetic",           "Energetic"),
    ("pump up",             "Energetic"),
    ("party",               "Dance / Party"),
    ("dance pop",           "Dance / Pop"),
    ("dance",               "Dance / Party"),
    ("club",                "Dance / Party"),
    ("focus",               "Focus"),
    ("study",               "Focus"),
    ("concentration",       "Focus"),
    ("instrumental",        "Focus"),
    ("happy",               "Happy"),
    ("feel good",           "Happy"),
    ("upbeat",              "Happy"),
    ("uplifting",           "Happy"),
    # Genre-based rules
    ("hip hop",            "Hip-Hop"),
    ("trap",               "Hip-Hop"),
    ("drill",              "Hip-Hop"),
    ("rap",                "Hip-Hop"),
    ("neo soul",           "R&B / Soul"),
    ("r&b",                "R&B / Soul"),
    ("soul",               "R&B / Soul"),
    ("edm",                "Electronic / Dance"),
    ("techno",             "Electronic / Dance"),
    ("house",              "Electronic / Dance"),
    ("electronic",         "Electronic / Dance"),
    ("indie pop",          "Indie / Alternative"),
    ("dream pop",          "Indie / Alternative"),
    ("k-pop",              "Pop"),
    ("pop",                "Pop"),
    ("shoegaze",           "Indie / Alternative"),
    ("indie rock",         "Indie / Alternative"),
    ("indie",              "Indie / Alternative"),
    ("alternative",        "Indie / Alternative"),
    ("hard rock",          "Rock"),
    ("classic rock",       "Rock"),
    ("rock",               "Rock"),
    ("metal",              "Metal"),
    ("lo-fi",              "Chill / Lo-Fi"),
    ("orchestral",         "Classical"),
    ("classical",          "Classical"),
    ("jazz",               "Jazz / Blues"),
    ("blues",              "Jazz / Blues"),
    ("country",            "Country / Folk"),
    ("folk",               "Country / Folk"),
    ("singer-songwriter",  "Acoustic"),
    ("acoustic",           "Acoustic"),
    ("reggaeton",          "Latin"),
    ("latin",              "Latin"),
    ("hardcore",           "Punk / Hardcore"),
    ("punk",               "Punk / Hardcore"),
]


def build_feature_matrix(
    df: pd.DataFrame,
    top_n: int = GENRE_TOP_N,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """
    Build feature matrix: one-hot encoded top-N tags from merged genres + Last.fm tags.
    Empty-tag tracks are imputed with the 3 most common library tags at weight 0.1.
    Returns (X_scaled, fitted_scaler, feature_names).
    """
    all_tag_lists: list[list[str]] = []
    for _, row in df.iterrows():
        merged: set[str] = set()
        for col in ("genres", "lastfm_tags"):
            val = row.get(col) or ""
            merged.update(t.strip().lower() for t in val.split(",") if t.strip())
        all_tag_lists.append(sorted(merged))

    tag_counts = Counter(t for tags in all_tag_lists for t in tags)
    top_tags = [t for t, _ in tag_counts.most_common(top_n)]

    if not top_tags:
        raise ValueError(
            "No tag data found in your library. Sync your library first to populate genres."
        )

    tag_index = {t: i for i, t in enumerate(top_tags)}
    tag_matrix = np.zeros((len(df), len(top_tags)), dtype=float)

    empty_indices: list[int] = []
    for i, tags in enumerate(all_tag_lists):
        if tags:
            for t in tags:
                if t in tag_index:
                    tag_matrix[i, tag_index[t]] = 1.0
        else:
            empty_indices.append(i)

    if empty_indices:
        impute_cols = [tag_index[t] for t in top_tags[:3] if t in tag_index]
        for i in empty_indices:
            for col_idx in impute_cols:
                tag_matrix[i, col_idx] = 0.1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tag_matrix)
    return X_scaled, scaler, top_tags


def select_k(
    X: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        raise ValueError(
            f"Need at least {k_min + 1} tracks to cluster; got {n}."
        )
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
        logger.debug("K=%d  silhouette=%.4f", k, scores[k])
    best_k = max(scores, key=scores.__getitem__)
    logger.info("Selected K=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(X)
    return km


def assign_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["cluster"] = labels.astype(int)
    return out


def infer_mood_label(top_genres: list[str]) -> str:
    """Derive a mood label by matching dominant genre names against _GENRE_MOOD_RULES."""
    for genre in top_genres:
        g = genre.lower()
        for key, label in _GENRE_MOOD_RULES:
            if key in g:
                return label
    return "Mixed"


def save_clusters(
    session: Session,
    user: User,
    kmeans: KMeans,
    scaler: StandardScaler,
    feature_names: list[str],
) -> None:
    """Replace all cluster rows for this user. Labels from dominant centroid genres."""
    session.query(Cluster).filter_by(user_id=user.id).delete()

    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)

    for idx, raw in enumerate(centroids_orig):
        centroid = dict(zip(feature_names, raw))
        top_genres = sorted(centroid, key=centroid.__getitem__, reverse=True)[:5]

        session.add(Cluster(
            user_id=user.id,
            cluster_index=idx,
            label=infer_mood_label(top_genres),
        ))

    session.commit()
    logger.info("Saved %d cluster rows for user %d", len(centroids_orig), user.id)


def load_cluster_labels(session: Session, user: User) -> dict[int, str]:
    rows = session.query(Cluster).filter_by(user_id=user.id).all()
    return {row.cluster_index: (row.label or "") for row in rows}


def infer_current_mood(
    df_tracks: pd.DataFrame,
    df_history: pd.DataFrame,
    hour: int,
    window: int = 2,
) -> int:
    def _fallback() -> int:
        if not df_tracks.empty and "cluster" in df_tracks.columns:
            return int(df_tracks["cluster"].mode().iloc[0])
        return 0

    if df_history.empty or "cluster" not in df_tracks.columns:
        return _fallback()

    window_hours = {(hour + d) % 24 for d in range(-window, window + 1)}
    nearby = df_history[df_history["hour"].isin(window_hours)]
    if nearby.empty:
        return _fallback()

    merged = nearby.merge(df_tracks[["track_id", "cluster"]], on="track_id", how="inner")
    if merged.empty:
        return _fallback()

    return int(merged["cluster"].mode().iloc[0])


def pca_projection(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """Reduce scaled feature matrix to 2D via PCA for scatter visualisation."""
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    return pd.DataFrame({
        "name":    df["name"].values,
        "artist":  df["artist"].values,
        "cluster": df["cluster"].values,
        "pc1":     coords[:, 0],
        "pc2":     coords[:, 1],
    })


def genre_distribution(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Return the top N tags per cluster by track frequency.
    Merges genres and lastfm_tags per track using the same logic as build_feature_matrix,
    so the bar chart reflects what actually drove clustering.
    Output columns: cluster_index, genre, count.
    """
    records = []
    for cluster_idx, group in df.groupby("cluster"):
        all_tags: list[str] = []
        for _, row in group.iterrows():
            merged: set[str] = set()
            for col in ("genres", "lastfm_tags"):
                val = row.get(col) or ""
                merged.update(t.strip().lower() for t in val.split(",") if t.strip())
            all_tags.extend(merged)
        for tag, count in Counter(all_tags).most_common(top_n):
            records.append({"cluster_index": int(cluster_idx), "genre": tag, "count": count})
    if not records:
        return pd.DataFrame(columns=["cluster_index", "genre", "count"])
    return pd.DataFrame(records)


def run_clustering(
    df: pd.DataFrame,
    session: Session,
    user: User,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, dict[int, float]]:
    """Full pipeline: build features → select K → fit → assign → persist."""
    X, scaler, feature_names = build_feature_matrix(df)
    best_k, scores = select_k(X, k_min=k_min, k_max=k_max)
    kmeans = run_kmeans(X, best_k)
    df_clustered = assign_clusters(df, kmeans.labels_)
    save_clusters(session, user, kmeans, scaler, feature_names)
    for _, row in df_clustered.iterrows():
        session.query(Track).filter_by(id=int(row["track_id"])).update(
            {"cluster_index": int(row["cluster"])}
        )
    session.commit()
    return df_clustered, kmeans, scaler, scores
