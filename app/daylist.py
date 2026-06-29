from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import pandas as pd

_DOW_LABELS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_TIME_SLOTS: list[tuple[int, int, str]] = [
    (0,  4,  "late night"),
    (5,  8,  "early morning"),
    (9,  11, "morning"),
    (12, 14, "midday"),
    (15, 17, "afternoon"),
    (18, 20, "evening"),
    (21, 23, "night"),
]

# Ordered: first substring match wins. More specific phrases come before components.
_VIBE_RULES: list[tuple[str, str]] = [
    ("rainy day",    "rainy day"),
    ("heartbreak",   "heartbreak"),
    ("melancholic",  "melancholic"),
    ("dark",         "dark"),
    ("sad",          "sad"),
    ("chillhop",     "chillhop"),
    ("lo-fi",        "lo-fi"),
    ("chillout",     "chill"),
    ("chill",        "chill"),
    ("calm",         "calm"),
    ("peaceful",     "peaceful"),
    ("ambient",      "ambient"),
    ("workout",      "workout"),
    ("driving",      "driving"),
    ("energetic",    "energetic"),
    ("pump up",      "pump up"),
    ("party",        "party"),
    ("focus",        "focus"),
    ("study",        "study"),
    ("feel good",    "feel good"),
    ("upbeat",       "upbeat"),
    ("happy",        "happy"),
    ("nostalgic",    "nostalgic"),
    ("romantic",     "romantic"),
    ("summer",       "summer"),
    ("phonk",        "phonk"),
    ("dreamy",       "dreamy"),
]

# Collapse known Last.fm spelling variants to the canonical form used in _GENRE_RULES.
# Applied at score-accumulation time so both forms contribute to the same bucket.
_TAG_NORMALIZATIONS: dict[str, str] = {
    "hip-hop":  "hip hop",
    "neo soul": "neo-soul",
    "rnb":      "r&b",
    "kpop":     "k-pop",
}

# Genre rules: more specific multi-word entries before single-word components.
# Variants collapsed by _TAG_NORMALIZATIONS are omitted here.
_GENRE_RULES: list[str] = [
    "pop rap", "abstract hip-hop", "underground rap", "east coast hip-hop",
    "hip hop", "cloud rap", "trap", "drill", "rap",
    "neo-soul", "alternative rnb", "r&b", "soul", "funk",
    "indietronica", "bedroom pop", "dream pop", "indie pop", "indie rock",
    "shoegaze", "indie", "alternative",
    "k-rnb", "k-indie", "k-pop", "j-pop", "j-rock",
    "chillhop", "lo-fi",
    "edm", "techno", "house", "electronic", "dance",
    "pop",
    "jazz", "blues", "bossa nova",
    "classical", "orchestral",
    "folk", "singer-songwriter", "acoustic",
    "country",
    "hard rock", "classic rock", "rock",
    "metal", "hardcore", "punk",
    "reggaeton", "latin",
    "video game music",
]


def get_time_label(hour: int) -> str:
    for start, end, label in _TIME_SLOTS:
        if start <= hour <= end:
            return label
    return "night"


def _weighted_tag_scores(subset: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for _, row in subset.iterrows():
        w = float(row["weight"])
        for tag in (row["tags"] or "").split(","):
            tag = tag.strip().lower()
            if tag:
                scores[_TAG_NORMALIZATIONS.get(tag, tag)] += w
    return dict(scores)


def _relative_scores(
    window_scores: dict[str, float],
    global_scores: dict[str, float],
    min_window_weight: float = 1.0,
    epsilon: float = 1e-3,
) -> dict[str, float]:
    """
    Lift of each tag: its proportional share in this window vs. its global share.
    Tags below min_window_weight are excluded to suppress single-play noise.
    A lift > 1 means the tag is more concentrated here than in the overall library.
    """
    global_total = sum(global_scores.values()) or 1.0
    window_total = sum(window_scores.values()) or 1.0
    result: dict[str, float] = {}
    for tag, w in window_scores.items():
        if w < min_window_weight:
            continue
        g_prop = global_scores.get(tag, 0.0) / global_total
        w_prop = w / window_total
        result[tag] = w_prop / (g_prop + epsilon)
    return result


def _pick_vibe(top_tags: list[str]) -> str | None:
    for key, label in _VIBE_RULES:
        for t in top_tags:
            if key in t:
                return label
    return None


def _pick_genre(top_tags: list[str]) -> str | None:
    for genre in _GENRE_RULES:
        for t in top_tags:
            if genre == t or genre in t:
                return genre
    return None


def _build_descriptor(
    scores: dict[str, float],
    time_label: str,
    day_label: str,
    n_plays: int,
    n_total: int,
    rank_scores: dict[str, float] | None = None,
) -> dict:
    """
    rank_scores, if provided, determines tag ordering for vibe/genre selection.
    scores is always used for the display top_scores values.
    Falls back to scores for ordering when rank_scores is empty or None.
    """
    ordering = rank_scores if rank_scores else scores
    top_tags = sorted(ordering, key=ordering.__getitem__, reverse=True)[:20]
    vibe = _pick_vibe(top_tags)
    genre = _pick_genre(top_tags)

    parts: list[str] = []
    if vibe and vibe not in (genre or ""):
        parts.append(vibe)
    if genre:
        parts.append(genre)
    parts.append(day_label)
    parts.append(time_label)

    return {
        "descriptor": " ".join(parts),
        "vibe": vibe,
        "genre": genre,
        "time_label": time_label,
        "day_label": day_label,
        "top_tags": top_tags[:10],
        "top_scores": {t: round(scores.get(t, 0.0), 2) for t in top_tags[:10]},
        "n_plays": n_plays,
        "n_total": n_total,
    }


def compute_daylist(
    df: pd.DataFrame,
    months: int,
    half_life_days: float,
    target_hour: int,
    target_dow: int,
    hour_window: int = 2,
    _filtered: pd.DataFrame | None = None,
    _global_scores: dict[str, float] | None = None,
) -> dict:
    """
    Compute a daylist descriptor for the given time context from the raw scrobble df.

    df must have columns: days_ago, day_of_week, hour, tags.
    months and half_life_days are applied here (not pre-applied in the df).
    Falls back progressively if the specific time window has too few plays.

    _filtered and _global_scores can be passed in from build_week_grid to avoid
    redundant recomputation across the 168-call sweep.
    """
    time_label = get_time_label(target_hour)
    day_label = _DOW_LABELS[target_dow]

    _empty_subset = pd.DataFrame(columns=["played_at", "artist", "track_name", "tags", "weight"])
    empty = {
        "descriptor": f"{day_label} {time_label}",
        "vibe": None, "genre": None,
        "time_label": time_label, "day_label": day_label,
        "top_tags": [], "top_scores": {},
        "n_plays": 0, "n_total": 0,
        "subset": _empty_subset,
        "fallback_level": 0,
    }

    if df.empty:
        return empty

    if _filtered is not None:
        filtered = _filtered
        n_total = len(filtered)
        global_scores = _global_scores or {}
    else:
        filtered = df[df["days_ago"] <= months * 30].copy()
        n_total = len(filtered)
        if filtered.empty:
            return empty
        lam = math.log(2) / half_life_days
        filtered["weight"] = filtered["days_ago"].apply(lambda d: math.exp(-lam * d))
        global_scores = _weighted_tag_scores(filtered)

    if filtered.empty:
        return empty

    # Build hour window
    hours = {(target_hour + delta) % 24 for delta in range(-hour_window, hour_window + 1)}

    # Try: specific day + time window
    subset = filtered[filtered["hour"].isin(hours) & (filtered["day_of_week"] == target_dow)]
    fallback_level = 0

    # Fall back to same time window across all days if too sparse
    if len(subset) < 15:
        subset = filtered[filtered["hour"].isin(hours)]
        fallback_level = 1

    # Fall back to all data if still empty
    if subset.empty:
        subset = filtered
        fallback_level = 2

    scores = _weighted_tag_scores(subset)
    if not scores:
        return {**empty, "n_total": n_total, "subset": subset, "fallback_level": fallback_level}

    rel = _relative_scores(scores, global_scores) if global_scores else {}
    result = _build_descriptor(scores, time_label, day_label, len(subset), n_total, rank_scores=rel or None)
    result["subset"] = subset.sort_values("played_at", ascending=False).reset_index(drop=True)
    result["fallback_level"] = fallback_level
    return result


def build_week_grid(
    df: pd.DataFrame,
    months: int = 6,
    half_life_days: float = 30,
) -> pd.DataFrame:
    """Compute daylist for all 7 days x 24 hours; returns 168-row DataFrame."""
    # Pre-compute filtered+weighted df and global baseline once for all 168 calls.
    if df.empty:
        filtered = df.copy()
        global_scores: dict[str, float] = {}
    else:
        filtered = df[df["days_ago"] <= months * 30].copy()
        if not filtered.empty:
            lam = math.log(2) / half_life_days
            filtered["weight"] = filtered["days_ago"].apply(lambda d: math.exp(-lam * d))
            global_scores = _weighted_tag_scores(filtered)
        else:
            global_scores = {}

    rows = []
    for dow in range(7):
        for hour in range(24):
            r = compute_daylist(
                df, months, half_life_days, hour, dow,
                _filtered=filtered,
                _global_scores=global_scores,
            )
            rows.append({
                "day": _DOW_LABELS[dow].capitalize(),
                "dow": dow,
                "hour": hour,
                "time_slot": r["time_label"],
                "descriptor": r["descriptor"],
                "vibe": r["vibe"] or "",
                "genre": r["genre"] or "",
                "n_plays": r["n_plays"],
                "fallback": r["fallback_level"],
            })
    return pd.DataFrame(rows)
