"""
Master End-to-End Runner for PathSnap:
  Stage 1: Preprocessing, Dynamic UTM Projection, Candidate Generation & 37 Domain Features
  Stage 2: Leak-Free 2-Stage Feature Selection (Train 80 trips only)
  Stage 3: Train & Compare Best 5+ High-Accuracy Models (5-fold GroupKFold CV + Student-Test)
"""

import argparse
import time
from pathlib import Path

import config
from src.preprocessing import build_and_cache_datasets
from src.feature_selection import run_feature_selection
from src.train_models import run_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full PathSnap Preprocessing -> Feature Selection -> 5-Model Training pipeline.")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT,
                        help="Folder containing train/ (80 trips) and student_test/ (10 trips).")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Ignore cached candidate feature datasets and recompute from scratch.")
    args = parser.parse_args()

    t_all = time.perf_counter()

    print("=" * 75)
    print("STAGE 1: PREPROCESSING, CANDIDATE GENERATION & FEATURE ENGINEERING")
    print("=" * 75)
    train_df, test_df, _ = build_and_cache_datasets(args.data_root, force_rebuild=args.force_rebuild)

    print("\n" + "=" * 75)
    print("STAGE 2: LEAK-FREE FEATURE SELECTION (80 TRAINING TRIPS ONLY)")
    print("=" * 75)
    selected_features, _ = run_feature_selection(train_df)

    print("\n" + "=" * 75)
    print("STAGE 3: TRAINING & COMPARING THE BEST 5+ HIGH-ACCURACY MODELS")
    print("=" * 75)
    run_model_comparison(args.data_root, train_df, test_df, selected_features)

    print(f"\nAll 3 stages completed in {(time.perf_counter() - t_all) / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()
