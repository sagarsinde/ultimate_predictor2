"""
Walk-Forward Validation + Feature Ablation + Model Pruning
Saves intelligent weights and state to models/state.json
"""

import sys
import os

# Add the market directory to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.validator import (
    run_walk_forward, learn_weights, prune_models,
    build_calibration, run_feature_ablation, save_state,
)

MARKET_NAME = "TIME BAZAR"

def main():
    print(f"\n{'#'*70}")
    print(f"  SELF-IMPROVING ENSEMBLE — FULL BACKTEST: {MARKET_NAME}")
    print(f"{'#'*70}")

    # Step 1: Feature ablation
    print("\n[STEP 1/5] Feature Ablation...")
    surviving_groups = run_feature_ablation(verbose=True)

    # Step 2: Walk-forward validation with surviving features
    print("\n[STEP 2/5] Walk-Forward Validation with surviving features...")
    avg_metrics, (cal_data_m, cal_data_e) = run_walk_forward(
        active_groups=surviving_groups, verbose=True
    )

    if not avg_metrics:
        print("ERROR: No valid model metrics produced. Check your data.")
        sys.exit(1)

    # Step 3: Learn weights
    print("\n[STEP 3/5] Learning model weights from Brier Scores...")
    raw_weights = learn_weights(avg_metrics)

    print(f"\n  Raw weights (before pruning):")
    for mid, w in sorted(raw_weights.items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f}")

    # Step 4: Prune weak models
    print("\n[STEP 4/5] Pruning weak models (95% cumulative weight)...")
    pruned_weights = prune_models(raw_weights, cumulative_threshold=0.95)

    pruned_count = len(raw_weights) - len(pruned_weights)
    print(f"\n  Surviving: {len(pruned_weights)} models")
    print(f"  Pruned: {pruned_count} models")
    for mid, w in sorted(pruned_weights.items(), key=lambda x: -x[1]):
        print(f"    {mid:<15} {w:.4f} (renormalized)")

    # Step 5: Build confidence calibration
    print("\n[STEP 5/5] Building confidence calibration...")
    calibrator_m, thresholds_m = build_calibration(cal_data_m)
    calibrator_e, thresholds_e = build_calibration(cal_data_e)

    print(f"\n  Morning thresholds: {thresholds_m}")
    print(f"  Evening thresholds: {thresholds_e}")

    if calibrator_m is None:
        print("  WARNING: Not enough calibration data for morning. Using fallback thresholds.")
    if calibrator_e is None:
        print("  WARNING: Not enough calibration data for evening. Using fallback thresholds.")

    # Save everything
    save_state(
        pruned_weights, surviving_groups,
        calibrator_m, calibrator_e,
        thresholds_m, thresholds_e,
        avg_metrics,
    )

    print(f"\n{'#'*70}")
    print(f"  BACKTEST COMPLETE for {MARKET_NAME}")
    print(f"{'#'*70}\n")

if __name__ == '__main__':
    main()
