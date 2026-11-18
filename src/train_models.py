"""
Stage 3: Train & Compare the Best 5+ High-Accuracy Models Across Distinct Algorithm Families.

Models (calibrated natural class priors -> optimal Log Loss, Macro F1, Top-1 Accuracy & HMM emissions):
  1. LogisticRegression     (Linear & Probabilistic)
  2. RandomForest           (Bagging Ensemble)
  3. ExtraTrees             (Tree-based / Randomized Ensemble)
  4. LightGBM               (Boosting — Leaf-wise GBDT; auto-fallback to HistGradientBoosting)
  5. MLPNeuralNet           (Neural Networks — deep tabular MLP with authentic epoch-wise loss curves)
  6. XGBoost (if installed) (Boosting — Depth-wise Regularized GBDT)

Evaluation Protocol (strictly leak-free):
  - 5-fold GroupKFold CV (grouped by trip_id) on the 80 training trips.
  - Full refit on all 80 training trips; evaluation on the 10 student-testing trips.
  - Saves Report Table 3.1 CSV, fold-wise CSV, experiment configs JSON, comparison plots,
    confusion matrices, authentic MLP epoch curves, Folium map, and model+feature artifacts.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import load_partition, build_and_cache_datasets
from src.feature_selection import run_feature_selection
from src.metrics import evaluate_predictions, compute_candidate_metrics
from src.folium_viz import create_folium_map

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def build_best_models(seed: int = config.RANDOM_SEED) -> Dict[str, Dict[str, Any]]:
    """Build 5 (or 6 with XGBoost) strong, well-calibrated models across distinct families."""
    models: Dict[str, Dict[str, Any]] = {
        "LogisticRegression": {
            "family": "Linear & Probabilistic",
            "key_config": "C=1.0, lbfgs, StandardScaler, natural priors",
            "loss_objective": "L2-regularised Binary Cross-Entropy (Log Loss)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=seed,
                )),
            ]),
        },
        "RandomForest": {
            "family": "Bagging Ensemble",
            "key_config": "n_estimators=180, max_depth=22, min_samples_leaf=4, max_samples=0.45",
            "loss_objective": "Gini impurity reduction across bootstrapped trees",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=180,
                    max_depth=22,
                    min_samples_leaf=4,
                    max_samples=0.45,
                    n_jobs=-1,
                    random_state=seed,
                )),
            ]),
        },
        "ExtraTrees": {
            "family": "Tree-based (Randomized Ensemble)",
            "key_config": "n_estimators=180, max_depth=24, min_samples_leaf=4, bootstrap=True, max_samples=0.5",
            "loss_objective": "Gini impurity reduction with randomised split thresholds",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(
                    n_estimators=180,
                    max_depth=24,
                    min_samples_leaf=4,
                    bootstrap=True,
                    max_samples=0.5,
                    n_jobs=-1,
                    random_state=seed,
                )),
            ]),
        },
    }

    if HAS_LGBM:
        models["LightGBM"] = {
            "family": "Boosting (Leaf-wise GBDT)",
            "key_config": "n_estimators=500, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8, reg_lambda=1.0",
            "loss_objective": "Binary Log-Loss (leaf-wise gradient boosted decision trees)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", lgb.LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    min_child_samples=40,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=seed,
                    verbose=-1,
                )),
            ]),
        }
    else:
        models["HistGradientBoosting"] = {
            "family": "Boosting (Histogram GBDT)",
            "key_config": "max_iter=300, lr=0.06, max_leaf_nodes=63, l2=1.0",
            "loss_objective": "Binary Log-Loss (histogram gradient boosted trees)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=63,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=seed,
                )),
            ]),
        }

    models["MLPNeuralNet"] = {
        "family": "Neural Networks",
        "key_config": "hidden=(128,64,32), relu, adam, lr=1e-3, alpha=1e-3, batch=512",
        "loss_objective": "L2-regularised Binary Cross-Entropy (Adam optimiser)",
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                batch_size=512,
                max_iter=35,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=6,
                random_state=seed,
            )),
        ]),
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "family": "Boosting (Depth-wise GBDT)",
            "key_config": "n_estimators=500, lr=0.05, max_depth=8, subsample=0.8, colsample=0.8, hist",
            "loss_objective": "Binary Logistic Log-Loss (regularised depth-wise GBDT)",
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", xgb.XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    min_child_weight=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.5,
                    tree_method="hist",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=seed,
                )),
            ]),
        }

    return models


def record_mlp_epoch_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    max_epochs: int = 25,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Record genuine epoch-wise train & validation log-loss for MLPNeuralNet on a trip-grouped split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(X_train, y_train, groups))

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_tr = prep.fit_transform(X_train[tr_idx])
    X_va = prep.transform(X_train[va_idx])
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=512,
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )

    records = []
    for ep in range(1, max_epochs + 1):
        mlp.fit(X_tr, y_tr)
        p_tr = np.clip(mlp.predict_proba(X_tr)[:, 1], 1e-7, 1.0 - 1e-7)
        p_va = np.clip(mlp.predict_proba(X_va)[:, 1], 1e-7, 1.0 - 1e-7)
        records.append({
            "epoch": ep,
            "train_log_loss": float(log_loss(y_tr, p_tr, labels=[0, 1])),
            "val_log_loss": float(log_loss(y_va, p_va, labels=[0, 1])),
        })
    return pd.DataFrame(records)


def save_model_visualisations(
    summary_df: pd.DataFrame,
    cms: Dict[str, List[List[int]]],
    mlp_hist_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate all compulsory Section 3.3 visualisations from real experimental results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    models = summary_df["Model"].tolist()
    x = np.arange(len(models))

    # 1. Consolidated 4-panel model comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    axes[0, 0].bar(x, summary_df["CV Macro F1-score"], yerr=summary_df["CV Macro F1 Std"],
                   capsize=5, color="#2563eb", alpha=0.88)
    axes[0, 0].set_title("5-Fold GroupKFold CV Macro F1 (80 Train Trips)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=18, ha="right")
    axes[0, 0].set_ylim(max(0.0, float(summary_df["CV Macro F1-score"].min()) - 0.12), 1.02)
    axes[0, 0].grid(axis="y", alpha=0.3)

    w = 0.36
    axes[0, 1].bar(x - w / 2, summary_df["Top-1 Accuracy"], width=w, label="Top-1 Accuracy", color="#10b981")
    axes[0, 1].bar(x + w / 2, summary_df["Student-Test Macro F1-score"], width=w, label="Macro F1", color="#f59e0b")
    axes[0, 1].set_title("Student-Test Top-1 Accuracy & Macro F1 (10 Test Trips)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=18, ha="right")
    axes[0, 1].set_ylim(max(0.0, float(min(summary_df["Top-1 Accuracy"].min(), summary_df["Student-Test Macro F1-score"].min())) - 0.12), 1.02)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(x, summary_df["Log Loss"], color="#ef4444", alpha=0.85)
    axes[1, 0].set_title("Student-Test Log Loss (Lower is Better)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=18, ha="right")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(x, summary_df["Route Continuity"], color="#8b5cf6", alpha=0.85)
    axes[1, 1].set_title("Classifier-Only Route Continuity (Pre-HMM)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=18, ha="right")
    axes[1, 1].set_ylim(max(0.0, float(summary_df["Route Continuity"].min()) - 0.12), 1.02)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.suptitle("PathSnap Model Experimentation & Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_charts.png", dpi=160)
    plt.close()

    # 2. Confusion Matrices
    n_m = len(models)
    fig, axes = plt.subplots(1, n_m, figsize=(3.4 * n_m, 3.6))
    if n_m == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        cm = np.array(cms[name])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg (0)", "Pos (1)"])
        ax.set_yticklabels(["Neg (0)", "Pos (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8.5)
    plt.suptitle("Student-Test Candidate Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=160)
    plt.close()

    # 3. Authentic epoch-wise learning curve (MLPNeuralNet only — no artificial curves)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp_hist_df["epoch"], mlp_hist_df["train_log_loss"], marker="o", markersize=3, label="Training Log Loss")
    ax.plot(mlp_hist_df["epoch"], mlp_hist_df["val_log_loss"], marker="s", markersize=3, label="Validation Log Loss (Trip-Grouped)")
    ax.set_title("MLPNeuralNet Epoch-wise Training vs. Validation Log Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_training_curves.png", dpi=160)
    plt.close()


def run_model_comparison(
    data_root: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """Train & compare all 5+ models on the selected feature subset."""
    config.MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_trips = load_partition(data_root / "train", "train", load_route=True)
    test_trips = load_partition(data_root / "student_test", "student_test", load_route=True)
    train_by_id = {t.trip_id: t for t in train_trips}
    test_by_id = {t.trip_id: t for t in test_trips}

    X_train = train_df[selected_features].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    groups_train = train_df["trip_id"].to_numpy()
    X_test = test_df[selected_features].to_numpy(dtype=np.float32)

    suite = build_best_models(seed=config.RANDOM_SEED)
    gkf = GroupKFold(n_splits=config.N_CV_FOLDS)

    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    cms: Dict[str, List[List[int]]] = {}

    best_model_name = None
    best_cv_top1 = -1.0
    best_test_top1_df = None

    print(f"\n[Stage 3] Training {len(suite)} models on {len(selected_features)} selected features "
          f"({len(X_train):,} train rows, {len(X_test):,} student_test rows)...")

    for name, spec in suite.items():
        print(f"\n=== {name} ({spec['family']}) ===")
        cv_f1s, cv_top1s, cv_losses, cv_conts = [], [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
            m_fold = clone(spec["pipeline"])
            m_fold.fit(X_train[tr_idx], y_train[tr_idx])
            va_prob = m_fold.predict_proba(X_train[va_idx])[:, 1]
            va_df = train_df.iloc[va_idx].reset_index(drop=True)
            va_m, _ = evaluate_predictions(va_df, va_prob, train_by_id)

            cv_f1s.append(va_m["macro_f1"])
            cv_top1s.append(va_m["top1_accuracy"])
            cv_losses.append(va_m["log_loss"])
            cv_conts.append(va_m["route_continuity"])
            fold_rows.append({
                "Model": name,
                "Fold": fold_idx,
                "CV_Macro_F1": round(va_m["macro_f1"], 5),
                "CV_Top1_Accuracy": round(va_m["top1_accuracy"], 5),
                "CV_Log_Loss": round(va_m["log_loss"], 5),
                "CV_Route_Continuity": round(va_m["route_continuity"], 5),
            })

        # Full refit on all 80 training trips
        final_model = clone(spec["pipeline"])
        t_fit = time.perf_counter()
        final_model.fit(X_train, y_train)
        train_sec = time.perf_counter() - t_fit

        # Train-set diagnostic (to check overfitting vs underfitting)
        tr_prob = final_model.predict_proba(X_train)[:, 1]
        tr_m = compute_candidate_metrics(y_train, tr_prob)

        # Student-test evaluation
        t_inf = time.perf_counter()
        test_prob = final_model.predict_proba(X_test)[:, 1]
        inf_sec = time.perf_counter() - t_inf

        test_m, test_top1 = evaluate_predictions(test_df, test_prob, test_by_id)
        cms[name] = test_m["confusion_matrix"]

        # Save model + feature contract artifact
        joblib.dump(
            {"model_name": name, "features": selected_features, "pipeline": final_model},
            config.ARTIFACTS_DIR / f"{name}.joblib",
            compress=3,
        )

        row = {
            "Model": name,
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["key_config"],
            "CV Macro F1-score": round(float(np.mean(cv_f1s)), 4),
            "CV Macro F1 Std": round(float(np.std(cv_f1s)), 4),
            "CV Top-1 Accuracy": round(float(np.mean(cv_top1s)), 4),
            "Train Macro F1-score": round(tr_m["macro_f1"], 4),
            "Top-1 Accuracy": round(test_m["top1_accuracy"], 4),
            "Route Continuity": round(test_m["route_continuity"], 4),
            "Student-Test Macro F1-score": round(test_m["macro_f1"], 4),
            "Student-Test Precision": round(test_m["cand_precision"], 4),
            "Student-Test Recall": round(test_m["cand_recall"], 4),
            "Sequence Arc F1": round(test_m["sequence_arc_f1"], 4),
            "Log Loss": round(test_m["log_loss"], 4),
            "Training Time (s)": round(train_sec, 2),
            "Inference Time (s)": round(inf_sec, 4),
        }
        summary_rows.append(row)
        print(
            f"  CV Top-1: {row['CV Top-1 Accuracy']:.4f} | CV Macro F1: {row['CV Macro F1-score']:.4f} +/- {row['CV Macro F1 Std']:.4f} | "
            f"Train F1: {row['Train Macro F1-score']:.4f}\n"
            f"  Student-Test -> Top-1: {row['Top-1 Accuracy']:.4f} | Macro F1: {row['Student-Test Macro F1-score']:.4f} | "
            f"LogLoss: {row['Log Loss']:.4f} | Continuity: {row['Route Continuity']:.4f} | "
            f"SeqArcF1: {row['Sequence Arc F1']:.4f} | Inf: {row['Inference Time (s)']:.4f}s"
        )

        if float(np.mean(cv_top1s)) > best_cv_top1:
            best_cv_top1 = float(np.mean(cv_top1s))
            best_model_name = name
            best_test_top1_df = test_top1

    summary_df = pd.DataFrame(summary_rows).sort_values("Top-1 Accuracy", ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)

    summary_df.to_csv(config.MODEL_RESULTS_DIR / "model_comparison.csv", index=False)
    folds_df.to_csv(config.MODEL_RESULTS_DIR / "cv_fold_details.csv", index=False)

    print("\n[Stage 3] Recording authentic epoch-wise train/val loss curve for MLPNeuralNet...")
    mlp_hist_df = record_mlp_epoch_history(X_train, y_train, groups_train)
    mlp_hist_df.to_csv(config.MODEL_RESULTS_DIR / "mlp_epoch_history.csv", index=False)

    # Save experiment metadata
    meta = {
        "random_seed": config.RANDOM_SEED,
        "cv_strategy": f"{config.N_CV_FOLDS}-fold GroupKFold grouped by trip_id on 80 train trips",
        "selected_features": selected_features,
        "selected_best_model_by_cv_top1": best_model_name,
        "models": {
            k: {
                "family": v["family"],
                "key_config": v["key_config"],
                "loss_objective": v["loss_objective"],
            }
            for k, v in suite.items()
        },
        "confusion_matrices": cms,
    }
    with open(config.MODEL_RESULTS_DIR / "experiment_configs.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    save_model_visualisations(summary_df, cms, mlp_hist_df, config.MODEL_RESULTS_DIR)

    # Generate interactive Folium map comparing raw GPS, road network, reference route & best classifier predictions
    if best_test_top1_df is not None and len(test_trips) > 0:
        sample_trip = test_trips[0]
        pred_arcs = (
            best_test_top1_df.loc[best_test_top1_df["trip_id"] == sample_trip.trip_id, "arc_id"]
            .to_numpy(dtype=np.int64)
            .tolist()
        )
        create_folium_map(
            trip=sample_trip,
            out_html=config.MODEL_RESULTS_DIR / f"trip_{sample_trip.trip_id}_{best_model_name}_folium.html",
            predicted_arcs=pred_arcs,
            title=f"{best_model_name} (Pre-HMM)",
        )

    print("\n==================== FINAL MODEL COMPARISON (REPORT TABLE 3.1) ====================")
    cols = [
        "Model",
        "Algorithm Family",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route Continuity",
        "Student-Test Macro F1-score",
        "Log Loss",
        "Inference Time (s)",
    ]
    print(summary_df[cols].to_string(index=False))
    print(f"\nBest Model (selected via CV): {best_model_name}")
    print(f"Results saved to:   {config.MODEL_RESULTS_DIR}")
    print(f"Artifacts saved to: {config.ARTIFACTS_DIR}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 3 Model Training & Comparison.")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    args = parser.parse_args()

    train_df, test_df, _ = build_and_cache_datasets(args.data_root, force_rebuild=False)
    fs_path = config.FS_RESULTS_DIR / "selected_features.json"
    if fs_path.exists():
        with open(fs_path, "r", encoding="utf-8") as f:
            selected_features = json.load(f)["selected_features"]
    else:
        selected_features, _ = run_feature_selection(train_df)

    run_model_comparison(args.data_root, train_df, test_df, selected_features)
