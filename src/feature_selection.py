"""
Stage 2: Leak-Free 2-Stage Feature Selection (Fit Strictly on the 80 Training Trips).

Pipeline:
  1. Remove constant/zero-variance features.
  2. Compute Pearson correlation matrix on training partition; prune near-duplicate features
     with |r| > CORR_THRESHOLD, keeping the one with higher Mutual Information.
  3. Rank remaining features by consensus of:
       (a) Mutual Information (nonlinear univariate dependence)
       (b) LightGBM / HistGradientBoosting Gain & Permutation Importance (multivariate)
  4. Validate candidate subset sizes via 5-fold GroupKFold CV (grouped by trip_id) on train
     and pick the best-performing subset.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import ALL_ENGINEERED_FEATURES, build_and_cache_datasets

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _fast_selector_estimator(seed: int = config.RANDOM_SEED):
    if HAS_LGBM:
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.07,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=seed,
    )


def run_feature_selection(train_df: pd.DataFrame) -> Tuple[List[str], Dict]:
    """Perform leak-free feature selection strictly on the 80 training trips."""
    t0 = time.perf_counter()
    config.FS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features = [c for c in ALL_ENGINEERED_FEATURES if c in train_df.columns]
    X_all = train_df[features].to_numpy(dtype=np.float32)
    y_all = train_df["label"].to_numpy(dtype=np.int8)
    groups = train_df["trip_id"].to_numpy()

    # Subsample up to 100,000 rows from training trips for fast MI & correlation estimation
    rng = np.random.default_rng(config.RANDOM_SEED)
    sub_n = min(100_000, len(train_df))
    sub_idx = rng.choice(len(train_df), size=sub_n, replace=False)
    X_sub = np.nan_to_num(X_all[sub_idx], nan=0.0)
    y_sub = y_all[sub_idx]

    # Step 1: Drop zero-variance features
    stds = X_sub.std(axis=0)
    active_mask = stds > 1e-8
    dropped_const = [f for f, ok in zip(features, active_mask) if not ok]
    active_features = [f for f, ok in zip(features, active_mask) if ok]
    X_sub_act = X_sub[:, active_mask]

    # Step 2: Mutual Information on training subsample
    print(f"[Stage 2] Computing Mutual Information on {sub_n:,} training rows ({len(active_features)} features)...")
    mi_scores = mutual_info_classif(X_sub_act, y_sub, random_state=config.RANDOM_SEED, n_neighbors=5)
    mi_map = dict(zip(active_features, mi_scores))

    # Step 3: Collinearity pruning (|r| > CORR_THRESHOLD)
    corr_df = pd.DataFrame(X_sub_act, columns=active_features).corr().abs()
    dropped_corr_rows = []
    to_drop = set()
    for i in range(len(active_features)):
        fi = active_features[i]
        if fi in to_drop:
            continue
        for j in range(i + 1, len(active_features)):
            fj = active_features[j]
            if fj in to_drop:
                continue
            r_val = float(corr_df.iloc[i, j])
            if r_val > config.CORR_THRESHOLD:
                # Keep whichever has higher Mutual Information
                drop_feat, keep_feat = (fj, fi) if mi_map[fi] >= mi_map[fj] else (fi, fj)
                to_drop.add(drop_feat)
                dropped_corr_rows.append({
                    "dropped_feature": drop_feat,
                    "kept_feature": keep_feat,
                    "abs_pearson_r": round(r_val, 4),
                    "dropped_mi": round(float(mi_map[drop_feat]), 5),
                    "kept_mi": round(float(mi_map[keep_feat]), 5),
                })

    cand_features = [f for f in active_features if f not in to_drop]
    print(f"[Stage 2] Pruned {len(dropped_const)} constant & {len(to_drop)} collinear features -> {len(cand_features)} remaining.")

    # Step 4: Multivariate Importance on a trip-grouped train/val split inside train (80 trips)
    print("[Stage 2] Computing multivariate permutation + tree importance on trip-grouped split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
    tr_idx, va_idx = next(gss.split(train_df, y_all, groups))

    X_cand = train_df[cand_features].to_numpy(dtype=np.float32)
    est = _fast_selector_estimator(config.RANDOM_SEED)
    est.fit(X_cand[tr_idx], y_all[tr_idx])

    # Validation permutation importance (subsample validation to 40k for speed)
    va_sub = va_idx if len(va_idx) <= 40_000 else rng.choice(va_idx, size=40_000, replace=False)
    perm = permutation_importance(
        est,
        X_cand[va_sub],
        y_all[va_sub],
        scoring="neg_log_loss",
        n_repeats=3,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    perm_imp = np.maximum(perm.importances_mean, 0.0)

    if hasattr(est, "feature_importances_"):
        gain_imp = est.feature_importances_.astype(np.float64)
    else:
        gain_imp = perm_imp.copy()

    # Build consensus ranking (normalized MI + normalized Gain + normalized Permutation)
    mi_arr = np.array([mi_map[f] for f in cand_features], dtype=np.float64)
    def _norm(v: np.ndarray) -> np.ndarray:
        mx = v.max()
        return v / mx if mx > 1e-12 else np.zeros_like(v)

    consensus = 0.25 * _norm(mi_arr) + 0.35 * _norm(gain_imp) + 0.40 * _norm(perm_imp)

    rank_df = pd.DataFrame({
        "feature": cand_features,
        "mutual_info": np.round(mi_arr, 5),
        "tree_importance": np.round(gain_imp, 5),
        "perm_importance": np.round(perm_imp, 5),
        "consensus_score": np.round(consensus, 5),
    }).sort_values("consensus_score", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))

    # Step 5: Empirically validate feature subset sizes via 3-fold GroupKFold CV on train
    print("[Stage 2] Running GroupKFold CV ablation over top-K feature subsets...")
    candidate_ks = sorted(set([
        min(12, len(cand_features)),
        min(18, len(cand_features)),
        min(22, len(cand_features)),
        min(26, len(cand_features)),
        len(cand_features),
    ]))
    gkf = GroupKFold(n_splits=3)
    ablation_rows = []
    for k in candidate_ks:
        sel_cols = rank_df["feature"].iloc[:k].tolist()
        Xk = train_df[sel_cols].to_numpy(dtype=np.float32)
        f1s, losses, top1s = [], [], []
        for f_tr, f_va in gkf.split(Xk, y_all, groups):
            m = _fast_selector_estimator(config.RANDOM_SEED)
            m.fit(Xk[f_tr], y_all[f_tr])
            p = np.clip(m.predict_proba(Xk[f_va])[:, 1], 1e-7, 1.0 - 1e-7)
            f1s.append(f1_score(y_all[f_va], (p >= 0.5).astype(np.int8), average="macro"))
            losses.append(log_loss(y_all[f_va], p, labels=[0, 1]))
            # Quick observation Top-1 accuracy on validation fold
            va_meta = train_df.iloc[f_va][["trip_id", "obs_idx", "arc_id", "true_arc_id"]].copy()
            va_meta["prob"] = p
            best_rows = va_meta.loc[va_meta.groupby(["trip_id", "obs_idx"], sort=False)["prob"].idxmax()]
            top1s.append(float((best_rows["arc_id"] == best_rows["true_arc_id"]).mean()))

        ablation_rows.append({
            "n_features": k,
            "cv_macro_f1": round(float(np.mean(f1s)), 5),
            "cv_top1_accuracy": round(float(np.mean(top1s)), 5),
            "cv_log_loss": round(float(np.mean(losses)), 5),
        })
        print(f"  Top-{k:2d} features -> CV Top-1: {ablation_rows[-1]['cv_top1_accuracy']:.5f} | "
              f"CV Macro F1: {ablation_rows[-1]['cv_macro_f1']:.5f} | CV LogLoss: {ablation_rows[-1]['cv_log_loss']:.5f}")

    ablation_df = pd.DataFrame(ablation_rows)
    # Pick subset achieving best CV Log Loss (within 0.0005 of best Top-1, favoring cleaner subset)
    best_row = ablation_df.sort_values(["cv_top1_accuracy", "cv_log_loss"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["n_features"])
    selected_features = rank_df["feature"].iloc[:best_k].tolist()

    # Save tables & JSON
    rank_df.to_csv(config.FS_RESULTS_DIR / "feature_ranking.csv", index=False)
    pd.DataFrame(dropped_corr_rows).to_csv(config.FS_RESULTS_DIR / "dropped_collinear_features.csv", index=False)
    ablation_df.to_csv(config.FS_RESULTS_DIR / "subset_cv_ablation.csv", index=False)

    summary = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "partition_used": "train (80 trips only — strictly leak-free)",
        "initial_feature_count": len(features),
        "dropped_constant": dropped_const,
        "dropped_collinear_count": len(to_drop),
        "dropped_collinear": sorted(to_drop),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "best_subset_cv": best_row.to_dict(),
    }
    with open(config.FS_RESULTS_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots for Report Section 2.2
    _save_fs_plots(rank_df, ablation_df, corr_df, selected_features, config.FS_RESULTS_DIR)

    print(f"[Stage 2] Selected {len(selected_features)} features in {summary['elapsed_seconds']}s.")
    return selected_features, summary


def _save_fs_plots(
    rank_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    selected_features: List[str],
    out_dir: Path,
) -> None:
    # 1. Feature Importance & Subset Ablation Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    top_plot = rank_df.head(min(25, len(rank_df))).iloc[::-1]
    colors = ["#10b981" if f in selected_features else "#94a3b8" for f in top_plot["feature"]]
    axes[0].barh(top_plot["feature"], top_plot["consensus_score"], color=colors)
    axes[0].set_title("Top Candidate Features (Green = Selected)")
    axes[0].set_xlabel("Consensus Score (MI + Tree Gain + Permutation)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].plot(ablation_df["n_features"], ablation_df["cv_top1_accuracy"], marker="o", linewidth=2, label="CV Top-1 Accuracy", color="#2563eb")
    axes[1].plot(ablation_df["n_features"], ablation_df["cv_macro_f1"], marker="s", linewidth=2, label="CV Macro F1", color="#f59e0b")
    axes[1].axvline(len(selected_features), linestyle="--", color="#10b981", label=f"Selected (K={len(selected_features)})")
    axes[1].set_title("GroupKFold CV Performance vs. Feature Subset Size")
    axes[1].set_xlabel("Number of Selected Features")
    axes[1].set_ylabel("Score (Train 80-Trip CV)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("PathSnap Leak-Free Feature Selection (80 Training Trips Only)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_selection_summary.png", dpi=160)
    plt.close()

    # 2. Correlation heatmap of selected features
    sub_corr = corr_df.loc[selected_features, selected_features]
    plt.figure(figsize=(11, 9))
    sns.heatmap(sub_corr, cmap="coolwarm", vmin=0.0, vmax=1.0, square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Absolute Correlation Matrix of {len(selected_features)} Selected Features (Train Partition)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Leak-Free Feature Selection.")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    args = parser.parse_args()
    train_df, _, _ = build_and_cache_datasets(args.data_root, force_rebuild=False)
    run_feature_selection(train_df)
