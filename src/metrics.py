"""Evaluation metrics for candidate classification & map-matching route quality."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from src.preprocessing import TripData


def compute_candidate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    y_pred = (y_prob >= threshold).astype(np.int8)
    clipped = np.clip(y_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "cand_accuracy": float(accuracy_score(y_true, y_pred)),
        "cand_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cand_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pos_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def predict_top1_per_observation(df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    """Fast vectorized argmax per (trip_id, obs_idx) selecting highest-probability candidate arc."""
    work = df[["trip_id", "obs_idx", "arc_id", "true_arc_id"]].copy()
    work["prob"] = y_prob
    # Sort descending by prob within each observation and keep first
    top1 = (
        work.sort_values(["trip_id", "obs_idx", "prob"], ascending=[True, True, False], kind="mergesort")
        .drop_duplicates(subset=["trip_id", "obs_idx"], keep="first")
        .reset_index(drop=True)
    )
    return top1


def compute_route_metrics(
    top1_df: pd.DataFrame,
    trips_by_id: Dict[str, TripData],
) -> Dict[str, float]:
    """Compute observation-level Top-1 Accuracy, Route Continuity, and Sequence Arc F1."""
    top1_acc = float((top1_df["arc_id"] == top1_df["true_arc_id"]).mean())

    total_steps = 0
    valid_steps = 0
    seq_f1s: List[float] = []

    for trip_id, grp in top1_df.groupby("trip_id", sort=False):
        trip = trips_by_id[trip_id]
        from_arr = trip.arcs["from_node_id"].to_numpy(dtype=np.int64)
        to_arr = trip.arcs["to_node_id"].to_numpy(dtype=np.int64)
        pred_arcs = grp["arc_id"].to_numpy(dtype=np.int64)

        if len(pred_arcs) > 1:
            same = (pred_arcs[1:] == pred_arcs[:-1])
            connected = (to_arr[pred_arcs[:-1]] == from_arr[pred_arcs[1:]])
            total_steps += len(pred_arcs) - 1
            valid_steps += int(np.sum(same | connected))

        if trip.route is not None and len(trip.route) > 0:
            pred_set = set(pred_arcs.tolist())
            ref_set = set(trip.route.tolist())
            inter = len(pred_set & ref_set)
            if inter > 0:
                p = inter / len(pred_set)
                r = inter / len(ref_set)
                seq_f1s.append(2.0 * p * r / (p + r))
            else:
                seq_f1s.append(0.0)

    return {
        "top1_accuracy": top1_acc,
        "route_continuity": float(valid_steps / total_steps) if total_steps > 0 else 0.0,
        "sequence_arc_f1": float(np.mean(seq_f1s)) if seq_f1s else 0.0,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    trips_by_id: Dict[str, TripData],
    threshold: float = 0.5,
) -> Tuple[Dict, pd.DataFrame]:
    cand_m = compute_candidate_metrics(df["label"].to_numpy(dtype=np.int8), y_prob, threshold=threshold)
    top1_df = predict_top1_per_observation(df, y_prob)
    route_m = compute_route_metrics(top1_df, trips_by_id)
    return {**cand_m, **route_m}, top1_df
