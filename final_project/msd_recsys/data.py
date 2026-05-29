"""Data loading, filtering, splitting, and sparse-matrix construction for MSD.

The MSD challenge ships interactions as tab-separated triplets and track metadata
as a SQLite database. This module wraps the I/O + standard preprocessing so the
notebook stays clean.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_triplets(path: str | Path) -> pd.DataFrame:
    """Load any (user_id, song_id, play_count) triplet file.

    Works for train_triplets.txt, kaggle_visible_evaluation_triplets.txt, and
    year1_{valid,test}_triplets_hidden.txt — same TSV format.
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "song_id", "play_count"],
        dtype={"user_id": "string", "song_id": "string", "play_count": np.int32},
    )
    return df


def load_song_id_list(path: str | Path) -> list[str]:
    """Load kaggle_songs.txt — canonical list of song_ids.

    Each line pairs a song_id with its integer index, in EITHER order
    ("SOAAADD12A8C13D8C7 1" or "1 SOAAADD12A8C13D8C7"). We pick whichever
    token is non-numeric as the song_id, so column order doesn't matter — the
    old digit-on-first-token heuristic silently returned whole lines when the
    song_id came first, which broke downstream `song_id.isin(...)` matches.
    """
    song_ids: list[str] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                song_ids.append(parts[0])
                continue
            a, b = parts[0], parts[1]
            if a.isdigit() and not b.isdigit():
                song_ids.append(b)          # "1 SOxxx" -> SOxxx
            else:
                song_ids.append(a)          # "SOxxx 1" (or ambiguous) -> SOxxx
    return song_ids


def load_user_list(path: str | Path) -> list[str]:
    """Load kaggle_users.txt — canonical list of user_ids (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_song_index_map(path: str | Path) -> dict[str, int]:
    """Load kaggle_songs.txt as {song_id: integer_index}.

    The MSD Challenge submission lists each recommendation by its integer song
    index (1-based) from this file, not the raw song_id. The file has two
    whitespace-separated columns; this is order-agnostic — it detects which
    column is the integer index and which is the song_id.
    """
    mapping: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            a, b = parts
            if a.isdigit():
                song_id, idx = b, int(a)
            elif b.isdigit():
                song_id, idx = a, int(b)
            else:
                continue
            mapping[song_id] = idx
    return mapping


def load_song_to_track(path: str | Path) -> pd.DataFrame:
    """Load taste_profile_song_to_tracks.txt — song_id -> track_id mapping.

    Real format is variable-width:
        song_id<TAB>track_id_1[<TAB>track_id_2[<TAB>...]]
    because one song_id can map to multiple MSD track_ids (different versions
    of the same song). Returns long-format: one row per (song_id, track_id) pair.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            song_id = parts[0]
            for track_id in parts[1:]:
                if track_id:
                    rows.append((song_id, track_id))
    return pd.DataFrame(rows, columns=["song_id", "track_id"]).astype("string")


# Columns we expect from track_metadata.db. Confirm against your DB; the official
# MSD `songs` table has these names. If yours differs, edit METADATA_COLUMNS.
METADATA_COLUMNS = [
    "track_id", "title", "song_id", "release",
    "artist_id", "artist_name",
    "duration", "artist_familiarity", "artist_hotttnesss", "year",
]


def load_track_metadata(db_path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load track metadata from the SQLite DB into a DataFrame."""
    cols = columns or METADATA_COLUMNS
    col_sql = ", ".join(cols)
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(f"SELECT {col_sql} FROM songs", conn)
    # MSD encodes "missing" as 0 for numeric columns
    for c in ("artist_familiarity", "artist_hotttnesss", "year"):
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan
    return df


# ---------------------------------------------------------------------------
# Filtering & splitting
# ---------------------------------------------------------------------------

def filter_interactions(
    df: pd.DataFrame,
    *,
    min_song_listens: int = 50,
    min_user_listens: int = 20,
    max_passes: int = 3,
) -> pd.DataFrame:
    """Drop rare songs and rare users iteratively.

    Filtering one dimension shrinks the other (a song with 60 listens may drop
    below threshold once we remove the users we filtered). `max_passes` iterations
    is almost always enough to reach a fixed point on MSD.
    """
    out = df
    for i in range(max_passes):
        before = len(out)
        song_counts = out.groupby("song_id").size()
        out = out[out.song_id.isin(song_counts[song_counts >= min_song_listens].index)]
        user_counts = out.groupby("user_id").size()
        out = out[out.user_id.isin(user_counts[user_counts >= min_user_listens].index)]
        if len(out) == before:
            break
    return out.reset_index(drop=True)


def holdout_split(
    df: pd.DataFrame,
    *,
    n_per_user: int = 5,
    min_train_after_holdout: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's last `n_per_user` interactions for validation.

    MSD triplets don't have timestamps. We use the row order in `df` as a proxy
    for "later" — fine if your input is ordered by user_id then by the listen
    order encoded in the original file. If you have access to a timestamp,
    sort by it before calling this.

    Users with too-short history (<n_per_user + min_train_after_holdout) keep
    all their data in `train_inner` and contribute no valid rows.
    """
    sizes = df.groupby("user_id").size()
    eligible = sizes[sizes >= n_per_user + min_train_after_holdout].index

    df_sorted = df.copy()
    df_sorted["__rank_desc"] = df_sorted.groupby("user_id").cumcount(ascending=False)
    is_valid = df_sorted.user_id.isin(eligible) & (df_sorted.__rank_desc < n_per_user)

    valid = df_sorted[is_valid].drop(columns="__rank_desc").reset_index(drop=True)
    train_inner = df_sorted[~is_valid].drop(columns="__rank_desc").reset_index(drop=True)
    return train_inner, valid


# ---------------------------------------------------------------------------
# Sparse user-item matrix
# ---------------------------------------------------------------------------

def build_user_item_matrix(
    df: pd.DataFrame,
    *,
    confidence_alpha: float = 40.0,
    use_logged_confidence: bool = True,
) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
    """Build sparse user x item matrix for ALS.

    For implicit feedback, ALS expects confidence values, not raw counts.
    Standard transform (Hu, Koren, Volinsky 2008): c_ui = 1 + alpha * log(1 + r_ui)
    when use_logged_confidence else c_ui = 1 + alpha * r_ui.

    Returns:
        ui: sparse user x item CSR matrix.
        user_to_ix: dict mapping user_id -> row index.
        item_to_ix: dict mapping song_id -> col index.
    """
    users = df.user_id.unique()
    items = df.song_id.unique()
    user_to_ix = {u: i for i, u in enumerate(users)}
    item_to_ix = {it: i for i, it in enumerate(items)}

    rows = df.user_id.map(user_to_ix).values
    cols = df.song_id.map(item_to_ix).values
    counts = df.play_count.values.astype(np.float32)

    if use_logged_confidence:
        confidence = 1.0 + confidence_alpha * np.log1p(counts)
    else:
        confidence = 1.0 + confidence_alpha * counts

    ui = csr_matrix(
        (confidence.astype(np.float32), (rows, cols)),
        shape=(len(users), len(items)),
    )
    return ui, user_to_ix, item_to_ix


def histories_from_df(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {user_id: [song_id, ...]} from an interaction frame."""
    return df.groupby("user_id")["song_id"].apply(list).to_dict()


# ---------------------------------------------------------------------------
# Official MSD Challenge evaluation split
# ---------------------------------------------------------------------------

def load_eval_user_inputs(
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    restrict_users_to: set[str] | None = None,
) -> tuple[list[str], list[list[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Turn an official visible/hidden triplet pair into the structures the
    retrieval + eval stages expect.

    The MSD Challenge gives every evaluation user a *visible* slice of their
    listening history (the model's input) and a *hidden* slice (the held-out
    ground truth scored by MAP@500). Unlike `holdout_split`, this is the real
    challenge split — using it makes our MAP@500 comparable to the leaderboard.
    The mapping is:

        histories[i]    = visible songs for users[i]   -> retrieval profile + features
        owned[users[i]] = same visible songs           -> exclude_owned at rank time
        truth[users[i]] = hidden songs                 -> MAP@500 ground truth

    Only users present in BOTH files are kept: a user with no hidden plays can't
    be scored, and a user with no visible plays has an empty retrieval profile.

    Args:
        visible: triplets from year1_{valid,test}_triplets_visible.txt
        hidden:  triplets from year1_{valid,test}_triplets_hidden.txt
        restrict_users_to: optional set of user_ids to keep — e.g. the users that
            actually have a row (and therefore factors) in the ALS matrix. Pass
            `set(user_to_ix)` to drop users ALS can't score.

    Returns:
        users:     list[user_id], stable-sorted.
        histories: list[list[song_id]] aligned with `users` (visible plays).
        owned:     {user_id: set[song_id]} visible plays, for exclude_owned.
        truth:     {user_id: set[song_id]} hidden plays, for MAP@500.
    """
    vis_by_user = visible.groupby("user_id")["song_id"].apply(list)
    hid_by_user = hidden.groupby("user_id")["song_id"].apply(set)

    common = set(vis_by_user.index) & set(hid_by_user.index)
    if restrict_users_to is not None:
        common &= restrict_users_to
    users = sorted(common)

    histories = [list(vis_by_user[u]) for u in users]
    owned = {u: set(vis_by_user[u]) for u in users}
    truth = {u: set(hid_by_user[u]) for u in users}
    return users, histories, owned, truth
