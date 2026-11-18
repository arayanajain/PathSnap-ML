"""
Stage 1: Data Loading, Cleaning, Dynamic UTM Projection, Fast Candidate Generation,
Monotonic DP Alignment, and High-Signal Feature Engineering.

Designed for PathSnap's actual scale (up to 307k road arcs & 19k GPS points per trip).
Strictly leak-free: zero ground-truth data ever enters candidate generation or feature computation.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


ALL_ENGINEERED_FEATURES: List[str] = [
    # Spatial proximity & within-observation relative ranking
    "dist_to_arc",
    "dist_rank",
    "dist_diff_from_min",
    "dist_ratio_to_min",
    "dist_zscore_in_obs",
    "gaussian_dist_share",
    # Segment geometry & projection
    "proj_fraction",
    "is_interior_proj",
    "dist_to_from_node",
    "dist_to_to_node",
    "arc_length",
    # Directional alignment (critical for opposing two-way directed arcs)
    "heading_diff_deg",
    "cos_heading_diff",
    "heading_diff_rank",
    "heading_diff_from_min",
    "combined_geo_score",
    "combined_geo_rank",
    # Multi-step temporal trajectory context (filters GPS noise & separates opposing lanes)
    "prev_dist_to_arc",
    "next_dist_to_arc",
    "window3_mean_dist",
    "window5_mean_dist",
    "progress_along_arc",
    # GPS kinematics & irregular gap indicators
    "gps_speed_mps",
    "dt_prev",
    "dt_next",
    "is_stationary",
    "is_large_gap",
    # Local graph topology & candidate-chain continuity (leak-free!)
    "num_candidates",
    "from_out_degree",
    "to_in_degree",
    "to_out_degree",
    "has_prev_conn",
    "has_next_conn",
    "same_arc_in_prev",
    "same_arc_in_next",
    "best_prev_conn_dist",
    "best_next_conn_dist",
]


@dataclass
class TripData:
    trip_id: str
    partition: str
    trip_dir: Path
    track: pd.DataFrame          # lon, lat, timestamp
    nodes: pd.DataFrame          # node_id, lon, lat
    arcs: pd.DataFrame           # arc_id, from_node_id, to_node_id
    route: Optional[np.ndarray]  # 1D int64 array of reference arc_ids


class LocalMetricProjector:
    """Projects WGS84 (lon, lat) dynamically to each trip's local UTM zone (meters)."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        zone = max(1, min(60, int((self.ref_lon + 180.0) // 6.0) + 1))
        self.epsg = (32600 if self.ref_lat >= 0 else 32700) + zone
        self._transformer = (
            Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
            if HAS_PYPROJ
            else None
        )
        if self._transformer is None:
            a, e2 = 6378137.0, 0.00669437999014
            lat_rad = np.deg2rad(self.ref_lat)
            denom = np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
            self._m_lat = a * (1.0 - e2) / (denom ** 3) * (np.pi / 180.0)
            self._m_lon = (a / denom) * np.cos(lat_rad) * (np.pi / 180.0)

    def to_meters(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if self._transformer is not None:
            x, y = self._transformer.transform(lon, lat)
            return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return (lon - self.ref_lon) * self._m_lon, (lat - self.ref_lat) * self._m_lat


def _find_file(trip_dir: Path, ext: str) -> Optional[Path]:
    hits = sorted(trip_dir.glob(f"*{ext}"))
    return hits[0] if hits else None


def load_trip(trip_dir: Path, partition: str, load_route: bool = True) -> TripData:
    trip_dir = Path(trip_dir)
    track_p = _find_file(trip_dir, ".track")
    nodes_p = _find_file(trip_dir, ".nodes")
    arcs_p = _find_file(trip_dir, ".arcs")
    route_p = _find_file(trip_dir, ".route") if load_route else None

    if not (track_p and nodes_p and arcs_p):
        raise FileNotFoundError(f"Missing raw files in {trip_dir}")

    track = pd.read_csv(track_p, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    for c in ["lon", "lat", "timestamp"]:
        track[c] = pd.to_numeric(track[c], errors="coerce")

    nodes = pd.read_csv(nodes_p, sep=r"\s+", header=None, names=["lon", "lat"])
    nodes["lon"] = pd.to_numeric(nodes["lon"], errors="coerce")
    nodes["lat"] = pd.to_numeric(nodes["lat"], errors="coerce")
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))

    arcs = pd.read_csv(arcs_p, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    arcs["from_node_id"] = pd.to_numeric(arcs["from_node_id"], errors="coerce").astype(np.int64)
    arcs["to_node_id"] = pd.to_numeric(arcs["to_node_id"], errors="coerce").astype(np.int64)
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))

    route = None
    if route_p is not None and route_p.stat().st_size > 0:
        r_df = pd.read_csv(route_p, sep=r"\s+", header=None, names=["arc_id"])
        route = pd.to_numeric(r_df["arc_id"], errors="coerce").dropna().astype(np.int64).to_numpy()

    return TripData(
        trip_id=trip_dir.name,
        partition=partition,
        trip_dir=trip_dir,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def load_partition(partition_dir: Path, partition_name: str, load_route: bool = True) -> List[TripData]:
    dirs = sorted([p for p in Path(partition_dir).iterdir() if p.is_dir()])
    return [load_trip(d, partition=partition_name, load_route=load_route) for d in dirs]


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid/NaN rows, enforce chronological ordering, and remove duplicate timestamps."""
    mask = (
        track["lon"].notna()
        & track["lat"].notna()
        & track["timestamp"].notna()
        & track["lon"].between(-180.0, 180.0)
        & track["lat"].between(-90.0, 90.0)
    )
    df = track.loc[mask].sort_values("timestamp", kind="mergesort")
    return df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def point_to_segments_distance(
    px: np.ndarray,
    py: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized perpendicular distance (m) and clamped projection fraction u in [0, 1]."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(len_sq > 1e-12, len_sq, 1.0)
    u = ((px - x1) * dx + (py - y1) * dy) / safe_len_sq
    u = np.where(len_sq > 1e-12, np.clip(u, 0.0, 1.0), 0.0)
    dist = np.hypot(px - (x1 + u * dx), py - (y1 + u * dy))
    return dist, u


def compute_bearing_deg(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.mod(np.rad2deg(np.arctan2(x2 - x1, y2 - y1)), 360.0)


def angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def build_local_arc_kdtree(
    local_arc_ids: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    step_m: float = config.ARC_SAMPLE_STEP_M,
) -> Tuple[cKDTree, np.ndarray]:
    """Vectorized densified cKDTree over local road segments (zero Python loop over arcs)."""
    lx1, ly1 = x1[local_arc_ids], y1[local_arc_ids]
    lx2, ly2 = x2[local_arc_ids], y2[local_arc_ids]
    lengths = np.hypot(lx2 - lx1, ly2 - ly1)

    # Number of evenly spaced sample points along each segment (at least 2: start, end; midpoint for medium)
    counts = np.maximum(2, np.ceil(lengths / step_m).astype(np.int64) + 1)
    total_pts = int(counts.sum())

    # Vectorized segment expansion via np.repeat + intra-segment fractional offset t in [0, 1]
    seg_idx = np.repeat(np.arange(len(local_arc_ids), dtype=np.int64), counts)
    offsets = np.cumsum(counts) - counts
    within_idx = np.arange(total_pts, dtype=np.int64) - np.repeat(offsets, counts)
    denom = np.repeat(np.maximum(counts - 1, 1), counts)
    t = within_idx / denom

    sx = lx1[seg_idx] + t * (lx2[seg_idx] - lx1[seg_idx])
    sy = ly1[seg_idx] + t * (ly2[seg_idx] - ly1[seg_idx])
    sample_arc_ids = local_arc_ids[seg_idx]

    return cKDTree(np.column_stack([sx, sy])), sample_arc_ids


def align_track_to_route_fast(
    tx: np.ndarray,
    ty: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    route_arcs: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Monotonic Dynamic Programming aligning N GPS points to M chronological `.route` arcs.
    Minimizes sum_t dist(p_t, route[j_t]) subject to 0 <= j_0 <= ... <= j_{N-1} < M.
    """
    n_obs = len(tx)
    m_route = len(route_arcs)
    if m_route == 0:
        return np.full(n_obs, -1, dtype=np.int64)
    if m_route == 1:
        return np.full(n_obs, route_arcs[0], dtype=np.int64)

    rx1, ry1 = x1[route_arcs], y1[route_arcs]
    rx2, ry2 = x2[route_arcs], y2[route_arcs]

    # Compute N x M cost matrix in vectorized batches (memory-safe even for 20k x 2.3k)
    cost = np.empty((n_obs, m_route), dtype=np.float32)
    batch = 512
    for start in range(0, n_obs, batch):
        end = min(n_obs, start + batch)
        d, _ = point_to_segments_distance(
            tx[start:end, None],
            ty[start:end, None],
            rx1[None, :],
            ry1[None, :],
            rx2[None, :],
            ry2[None, :],
        )
        cost[start:end] = d.astype(np.float32)

    # Vectorized forward pass using np.minimum.accumulate
    dp_prev = cost[0].astype(np.float64)
    back = np.empty((n_obs, m_route), dtype=np.int32)
    idx_arange = np.arange(m_route, dtype=np.int32)

    for t in range(1, n_obs):
        run_min = np.minimum.accumulate(dp_prev)
        best_k = np.maximum.accumulate(np.where(dp_prev == run_min, idx_arange, 0))
        dp_prev = cost[t] + run_min
        back[t] = best_k

    # Backtrack optimal monotonic alignment
    aligned_j = np.empty(n_obs, dtype=np.int32)
    j_cur = int(np.argmin(dp_prev))
    for t in range(n_obs - 1, -1, -1):
        aligned_j[t] = j_cur
        j_cur = back[t, j_cur]

    return route_arcs[aligned_j]


def extract_trip_features(
    trip: TripData,
    radius_m: float = config.CANDIDATE_RADIUS_M,
    max_k: int = config.MAX_CANDIDATES_K,
    include_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Preprocess one trip, generate top-K candidate arcs per GPS observation,
    align labels (if include_labels=True), and compute all 37 leak-free candidate features.
    """
    track = clean_track(trip.track)
    n_obs = len(track)
    if n_obs == 0:
        return pd.DataFrame(), {"n_obs": 0, "n_rows": 0, "n_missed": 0, "miss_rate": 0.0}

    # 1. Dynamic per-trip UTM metric projection
    proj = LocalMetricProjector(track["lon"].mean(), track["lat"].mean())
    tx, ty = proj.to_meters(track["lon"].to_numpy(), track["lat"].to_numpy())
    nx_m, ny_m = proj.to_meters(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())

    from_ids = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
    to_ids = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
    x1, y1 = nx_m[from_ids], ny_m[from_ids]
    x2, y2 = nx_m[to_ids], ny_m[to_ids]

    arc_lengths = np.hypot(x2 - x1, y2 - y1)
    arc_bearings = compute_bearing_deg(x1, y1, x2, y2)

    # Full-graph node degrees
    n_nodes = len(trip.nodes)
    out_deg = np.bincount(from_ids, minlength=n_nodes)
    in_deg = np.bincount(to_ids, minlength=n_nodes)

    # 2. Leak-free local bounding-box pre-filter (shrinks 300k-arc graphs by ~10-30x)
    buf = config.BBOX_BUFFER_M
    min_tx, max_tx = tx.min() - buf, tx.max() + buf
    min_ty, max_ty = ty.min() - buf, ty.max() + buf
    local_mask = (
        (np.maximum(x1, x2) >= min_tx)
        & (np.minimum(x1, x2) <= max_tx)
        & (np.maximum(y1, y2) >= min_ty)
        & (np.minimum(y1, y2) <= max_ty)
    )
    local_arc_ids = np.flatnonzero(local_mask)
    if len(local_arc_ids) == 0:
        local_arc_ids = np.arange(len(from_ids), dtype=np.int64)

    # 3. GPS kinematics & smoothed heading (respecting large GPS gaps from EDA)
    ts = track["timestamp"].to_numpy(dtype=np.float64)
    dt_prev = np.ones(n_obs, dtype=np.float64)
    dt_next = np.ones(n_obs, dtype=np.float64)
    if n_obs > 1:
        diffs = np.maximum(np.diff(ts), 1e-3)
        dt_prev[1:] = diffs
        dt_prev[0] = diffs[0]
        dt_next[:-1] = diffs
        dt_next[-1] = diffs[-1]

    gps_speed = np.zeros(n_obs, dtype=np.float64)
    gps_heading = np.zeros(n_obs, dtype=np.float64)
    for t in range(n_obs):
        # Use centered 2-step window unless separated by a large GPS gap
        t_lo = t - 1 if (t > 0 and dt_prev[t] <= config.LARGE_GAP_SEC) else t
        t_hi = t + 1 if (t < n_obs - 1 and dt_next[t] <= config.LARGE_GAP_SEC) else t
        if t_hi == t_lo:
            if t < n_obs - 1:
                t_hi = t + 1
            elif t > 0:
                t_lo = t - 1
        if t_hi > t_lo:
            step_d = np.hypot(tx[t_hi] - tx[t_lo], ty[t_hi] - ty[t_lo])
            step_dt = max(ts[t_hi] - ts[t_lo], 1e-3)
            gps_speed[t] = step_d / step_dt
            if step_d >= 1.0:
                gps_heading[t] = float(compute_bearing_deg(tx[t_lo], ty[t_lo], tx[t_hi], ty[t_hi]))
            elif t > 0:
                gps_heading[t] = gps_heading[t - 1]
    # Backfill initial stationary headings from first moving point
    moving_idx = np.flatnonzero(gps_speed >= config.STATIONARY_SPEED_MPS)
    if len(moving_idx) > 0 and moving_idx[0] > 0:
        gps_heading[: moving_idx[0]] = gps_heading[moving_idx[0]]

    # 4. Vectorized spatial candidate query via densified cKDTree
    tree, sample_arc_ids = build_local_arc_kdtree(local_arc_ids, x1, y1, x2, y2)
    q_k = min(64, len(sample_arc_ids))
    _, nn_indices = tree.query(np.column_stack([tx, ty]), k=q_k, workers=-1)
    if nn_indices.ndim == 1:
        nn_indices = nn_indices[:, None]

    # Ground-truth monotonic alignment (training / student-test only)
    aligned_true: Optional[np.ndarray] = None
    if include_labels and trip.route is not None and len(trip.route) > 0:
        aligned_true = align_track_to_route_fast(tx, ty, x1, y1, x2, y2, trip.route)

    # First pass: select top-K candidate arcs & distances per observation
    obs_cands: List[np.ndarray] = []
    obs_dists: List[np.ndarray] = []
    obs_us: List[np.ndarray] = []
    n_missed = 0

    for t in range(n_obs):
        raw_arcs = np.unique(sample_arc_ids[nn_indices[t]])
        dists, proj_u = point_to_segments_distance(
            tx[t], ty[t], x1[raw_arcs], y1[raw_arcs], x2[raw_arcs], y2[raw_arcs]
        )
        in_rad = dists <= radius_m
        if np.any(in_rad):
            raw_arcs = raw_arcs[in_rad]
            dists = dists[in_rad]
            proj_u = proj_u[in_rad]

        order = np.argsort(dists)[:max_k]
        c_arcs = raw_arcs[order]
        c_dists = dists[order]
        c_us = proj_u[order]

        obs_cands.append(c_arcs)
        obs_dists.append(c_dists)
        obs_us.append(c_us)

        if aligned_true is not None and aligned_true[t] not in c_arcs:
            n_missed += 1

    # Second pass: assemble flat NumPy arrays for all candidate rows (blazing fast!)
    counts = np.array([len(c) for c in obs_cands], dtype=np.int64)
    total_rows = int(counts.sum())

    obs_idx_col = np.repeat(np.arange(n_obs, dtype=np.int32), counts)
    arc_id_col = np.concatenate(obs_cands).astype(np.int64)
    dist_col = np.concatenate(obs_dists).astype(np.float32)
    u_col = np.concatenate(obs_us).astype(np.float32)

    # Pre-allocate per-row arrays for group-relative & neighbor-chain features
    dist_rank = np.empty(total_rows, dtype=np.float32)
    dist_diff_min = np.empty(total_rows, dtype=np.float32)
    dist_ratio_min = np.empty(total_rows, dtype=np.float32)
    dist_zscore = np.empty(total_rows, dtype=np.float32)
    gauss_share = np.empty(total_rows, dtype=np.float32)
    heading_diff = np.empty(total_rows, dtype=np.float32)
    heading_rank = np.empty(total_rows, dtype=np.float32)
    heading_diff_min = np.empty(total_rows, dtype=np.float32)
    comb_score = np.empty(total_rows, dtype=np.float32)
    comb_rank = np.empty(total_rows, dtype=np.float32)
    has_prev_conn = np.zeros(total_rows, dtype=np.float32)
    has_next_conn = np.zeros(total_rows, dtype=np.float32)
    same_prev = np.zeros(total_rows, dtype=np.float32)
    same_next = np.zeros(total_rows, dtype=np.float32)
    best_prev_conn_d = np.full(total_rows, radius_m * 1.25, dtype=np.float32)
    best_next_conn_d = np.full(total_rows, radius_m * 1.25, dtype=np.float32)

    # Precompute per-observation candidate sets for O(K^2) topological connectivity checks
    cursor = 0
    for t in range(n_obs):
        k = int(counts[t])
        sl = slice(cursor, cursor + k)
        c_arcs = obs_cands[t]
        c_d = obs_dists[t]

        min_d = c_d[0]
        mean_d = c_d.mean()
        std_d = c_d.std() + 1e-3

        dist_rank[sl] = np.arange(k, dtype=np.float32)
        dist_diff_min[sl] = c_d - min_d
        dist_ratio_min[sl] = (c_d + 1.0) / (min_d + 1.0)
        dist_zscore[sl] = (c_d - mean_d) / std_d

        g_w = np.exp(-0.5 * (c_d / 15.0) ** 2)
        gauss_share[sl] = g_w / (g_w.sum() + 1e-12)

        h_d = angular_diff_deg(gps_heading[t], arc_bearings[c_arcs])
        heading_diff[sl] = h_d
        heading_diff_min[sl] = h_d - h_d.min()
        heading_rank[sl] = np.argsort(np.argsort(h_d)).astype(np.float32)

        cs = c_d + 0.25 * h_d
        comb_score[sl] = cs
        comb_rank[sl] = np.argsort(np.argsort(cs)).astype(np.float32)

        # Leak-free predecessor/successor topological connectivity (ignoring across huge GPS gaps)
        if t > 0 and dt_prev[t] <= config.LARGE_GAP_SEC:
            p_arcs = obs_cands[t - 1]
            p_dists = obs_dists[t - 1]
            p_to = to_ids[p_arcs]
            c_from = from_ids[c_arcs]
            for i in range(k):
                a_i = c_arcs[i]
                same_mask = (p_arcs == a_i)
                conn_mask = same_mask | (p_to == c_from[i])
                if np.any(same_mask):
                    same_prev[cursor + i] = 1.0
                if np.any(conn_mask):
                    has_prev_conn[cursor + i] = 1.0
                    best_prev_conn_d[cursor + i] = float(p_dists[conn_mask].min())

        if t < n_obs - 1 and dt_next[t] <= config.LARGE_GAP_SEC:
            n_arcs = obs_cands[t + 1]
            n_dists = obs_dists[t + 1]
            n_from = from_ids[n_arcs]
            c_to = to_ids[c_arcs]
            for i in range(k):
                a_i = c_arcs[i]
                same_mask = (n_arcs == a_i)
                conn_mask = same_mask | (c_to[i] == n_from)
                if np.any(same_mask):
                    same_next[cursor + i] = 1.0
                if np.any(conn_mask):
                    has_next_conn[cursor + i] = 1.0
                    best_next_conn_d[cursor + i] = float(n_dists[conn_mask].min())

        cursor += k

    # Vectorized multi-step distances & progress_along_arc (u_{t+1} - u_{t-1})
    idx_m2 = np.clip(obs_idx_col - 2, 0, n_obs - 1)
    idx_m1 = np.clip(obs_idx_col - 1, 0, n_obs - 1)
    idx_p1 = np.clip(obs_idx_col + 1, 0, n_obs - 1)
    idx_p2 = np.clip(obs_idx_col + 2, 0, n_obs - 1)

    ax1, ay1 = x1[arc_id_col], y1[arc_id_col]
    ax2, ay2 = x2[arc_id_col], y2[arc_id_col]

    d_m2, _ = point_to_segments_distance(tx[idx_m2], ty[idx_m2], ax1, ay1, ax2, ay2)
    d_m1, u_m1 = point_to_segments_distance(tx[idx_m1], ty[idx_m1], ax1, ay1, ax2, ay2)
    d_p1, u_p1 = point_to_segments_distance(tx[idx_p1], ty[idx_p1], ax1, ay1, ax2, ay2)
    d_p2, _ = point_to_segments_distance(tx[idx_p2], ty[idx_p2], ax1, ay1, ax2, ay2)

    win3_dist = (d_m1 + dist_col + d_p1) / 3.0
    win5_dist = (d_m2 + d_m1 + dist_col + d_p1 + d_p2) / 5.0
    progress_u = (u_p1 - u_m1).astype(np.float32)

    # Assemble clean DataFrame (float32 features for fast low-RAM training)
    true_arc_col = (
        aligned_true[obs_idx_col] if aligned_true is not None else np.full(total_rows, -1, dtype=np.int64)
    )
    label_col = (
        (arc_id_col == true_arc_col).astype(np.int8)
        if aligned_true is not None
        else np.full(total_rows, -1, dtype=np.int8)
    )

    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": obs_idx_col,
        "arc_id": arc_id_col,
        "true_arc_id": true_arc_col,
        "label": label_col,
        "dist_to_arc": dist_col,
        "dist_rank": dist_rank,
        "dist_diff_from_min": dist_diff_min,
        "dist_ratio_to_min": dist_ratio_min,
        "dist_zscore_in_obs": dist_zscore,
        "gaussian_dist_share": gauss_share,
        "proj_fraction": u_col,
        "is_interior_proj": ((u_col > 1e-4) & (u_col < 1.0 - 1e-4)).astype(np.float32),
        "dist_to_from_node": np.hypot(tx[obs_idx_col] - ax1, ty[obs_idx_col] - ay1).astype(np.float32),
        "dist_to_to_node": np.hypot(tx[obs_idx_col] - ax2, ty[obs_idx_col] - ay2).astype(np.float32),
        "arc_length": arc_lengths[arc_id_col].astype(np.float32),
        "heading_diff_deg": heading_diff,
        "cos_heading_diff": np.cos(np.deg2rad(heading_diff)).astype(np.float32),
        "heading_diff_rank": heading_rank,
        "heading_diff_from_min": heading_diff_min,
        "combined_geo_score": comb_score,
        "combined_geo_rank": comb_rank,
        "prev_dist_to_arc": d_m1.astype(np.float32),
        "next_dist_to_arc": d_p1.astype(np.float32),
        "window3_mean_dist": win3_dist.astype(np.float32),
        "window5_mean_dist": win5_dist.astype(np.float32),
        "progress_along_arc": progress_u,
        "gps_speed_mps": gps_speed[obs_idx_col].astype(np.float32),
        "dt_prev": np.minimum(dt_prev[obs_idx_col], 300.0).astype(np.float32),
        "dt_next": np.minimum(dt_next[obs_idx_col], 300.0).astype(np.float32),
        "is_stationary": (gps_speed[obs_idx_col] < config.STATIONARY_SPEED_MPS).astype(np.float32),
        "is_large_gap": (dt_prev[obs_idx_col] > config.LARGE_GAP_SEC).astype(np.float32),
        "num_candidates": counts[obs_idx_col].astype(np.float32),
        "from_out_degree": out_deg[from_ids[arc_id_col]].astype(np.float32),
        "to_in_degree": in_deg[to_ids[arc_id_col]].astype(np.float32),
        "to_out_degree": out_deg[to_ids[arc_id_col]].astype(np.float32),
        "has_prev_conn": has_prev_conn,
        "has_next_conn": has_next_conn,
        "same_arc_in_prev": same_prev,
        "same_arc_in_next": same_next,
        "best_prev_conn_dist": best_prev_conn_d,
        "best_next_conn_dist": best_next_conn_d,
    })

    stats = {
        "trip_id": trip.trip_id,
        "n_obs": n_obs,
        "n_rows": total_rows,
        "n_local_arcs": int(len(local_arc_ids)),
        "n_total_arcs": int(len(from_ids)),
        "n_missed": int(n_missed),
        "miss_rate": float(n_missed / n_obs),
        "utm_epsg": int(proj.epsg),
    }
    return df, stats


def build_and_cache_datasets(data_root: Path, force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Run Stage 1 preprocessing on train (80) and student_test (10), and cache to disk."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.PREP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_cache = config.CACHE_DIR / "train_candidates.joblib"
    test_cache = config.CACHE_DIR / "student_test_candidates.joblib"
    stats_path = config.PREP_RESULTS_DIR / "preprocessing_summary.json"

    if not force_rebuild and train_cache.exists() and test_cache.exists() and stats_path.exists():
        print("[Stage 1] Loading cached preprocessed datasets...")
        train_df = joblib.load(train_cache)
        test_df = joblib.load(test_cache)
        with open(stats_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return train_df, test_df, summary

    t0 = time.perf_counter()
    train_trips = load_partition(data_root / "train", "train", load_route=True)
    test_trips = load_partition(data_root / "student_test", "student_test", load_route=True)
    print(f"[Stage 1] Loaded {len(train_trips)} train trips & {len(test_trips)} student_test trips.")

    def _process(trips: List[TripData], label: str):
        dfs, per_trip = [], []
        for idx, tr in enumerate(trips, 1):
            df_t, st = extract_trip_features(tr, include_labels=True)
            dfs.append(df_t)
            per_trip.append(st)
            if idx % 20 == 0 or idx == len(trips):
                print(f"  [{label}] {idx}/{len(trips)} trips processed...")
        full_df = pd.concat(dfs, ignore_index=True)
        tot_obs = sum(s["n_obs"] for s in per_trip)
        tot_miss = sum(s["n_missed"] for s in per_trip)
        part_summary = {
            "n_trips": len(trips),
            "total_observations": tot_obs,
            "total_candidate_rows": len(full_df),
            "positive_rate": round(float(full_df["label"].mean()), 5),
            "total_missed_observations": tot_miss,
            "candidate_miss_rate": round(tot_miss / max(tot_obs, 1), 5),
        }
        return full_df, part_summary, pd.DataFrame(per_trip)

    train_df, train_sum, train_trip_df = _process(train_trips, "train")
    test_df, test_sum, test_trip_df = _process(test_trips, "student_test")

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "elapsed_seconds": elapsed,
        "candidate_radius_m": config.CANDIDATE_RADIUS_M,
        "max_candidates_k": config.MAX_CANDIDATES_K,
        "bbox_buffer_m": config.BBOX_BUFFER_M,
        "n_engineered_features": len(ALL_ENGINEERED_FEATURES),
        "engineered_features": ALL_ENGINEERED_FEATURES,
        "train": train_sum,
        "student_test": test_sum,
    }

    joblib.dump(train_df, train_cache, compress=3)
    joblib.dump(test_df, test_cache, compress=3)
    train_trip_df.to_csv(config.PREP_RESULTS_DIR / "train_candidate_stats.csv", index=False)
    test_trip_df.to_csv(config.PREP_RESULTS_DIR / "student_test_candidate_stats.csv", index=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Stage 1] Done in {elapsed}s!")
    print(f"  Train: {train_sum}")
    print(f"  Student-Test: {test_sum}")
    return train_df, test_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1 Preprocessing & Feature Engineering.")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--force", action="store_true", help="Rebuild cached feature datasets.")
    args = parser.parse_args()
    build_and_cache_datasets(args.data_root, force_rebuild=args.force)
