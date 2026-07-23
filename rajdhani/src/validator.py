"""
v2/validator.py — Walk-Forward Validation, Weight Learning, Pruning & Calibration

The heart of the v2 system. This module:
  1. Runs rolling walk-forward validation across multiple periods
  2. Computes per-model metrics (Top-1, Top-3, Log Loss, Brier Score)
  3. Learns model weights from Brier Scores via softmax
  4. Prunes weak models (keep 95% cumulative weight)
  5. Runs feature ablation (optional, for initial setup)
  6. Builds confidence calibration from validation history
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict

from src.features import (
    load_raw_data, build_features, slice_window,
    get_window_size, ALL_FEATURE_GROUPS,
)
from src.models import MODEL_TYPES, FEATURE_MODELS, SEQUENCE_MODELS

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def _ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)


def _get_validation_periods(df, num_periods=5, pred_days=7):
    """
    Generate rolling monthly validation periods.
    Each period: train up to end of month M, predict first `pred_days`
    playing days of month M+1.

    Returns list of (train_end_date, pred_dates_list) tuples.
    """
    df_dates = pd.to_datetime(df['Date'])
    # Get unique months in the data
    months = sorted(df_dates.dt.to_period('M').unique())

    # We need at least 2 months of data, and we use the last num_periods months
    if len(months) < 2:
        raise ValueError("Not enough data for validation periods")

    # Use the last num_periods+1 months (train on M, predict M+1)
    # We need the prediction month to have actual data to compare against
    usable_months = months[-num_periods - 1:]

    periods = []
    for i in range(len(usable_months) - 1):
        train_month = usable_months[i]
        pred_month = usable_months[i + 1]

        train_end = train_month.end_time.date()

        # Get the first pred_days draws from the prediction month
        pred_mask = df_dates.dt.to_period('M') == pred_month
        pred_indices = df.index[pred_mask][:pred_days]

        if len(pred_indices) < 3:  # Need at least 3 days to evaluate
            continue

        periods.append((train_end, pred_indices.tolist()))

    return periods


def _train_single_model(model_type, window_label, train_df, active_groups):
    """
    Train a single model on a given data slice.

    Returns: (model_morning, model_evening, feature_columns)
    """
    window_draws = get_window_size(window_label)
    sliced_df = slice_window(train_df, window_draws)

    if len(sliced_df) < 10:
        return None, None, None

    feature_df_m, feature_df_e, y_m, y_e, group_cols = build_features(sliced_df, active_groups)

    # Drop internal date column for ML
    feature_cols_m = [c for c in feature_df_m.columns if c != '_date']
    X_m = feature_df_m[feature_cols_m].values

    feature_cols_e = [c for c in feature_df_e.columns if c != '_date']
    X_e = feature_df_e[feature_cols_e].values

    if len(X_m) < 5 or len(X_e) < 5:
        return None, None, None

    # Morning sequence and Evening sequence for Markov/Frequency
    m_seq = sliced_df['Morning_number'].astype(int).values
    e_seq = sliced_df['Evening_number'].astype(int).values
    dow_seq = pd.to_datetime(sliced_df['Date']).dt.dayofweek.values

    model_cls = MODEL_TYPES[model_type]

    # Train morning model
    model_m = model_cls()
    if model_type in FEATURE_MODELS:
        model_m.fit(X_m, y_m.values)
    else:
        model_m.fit(None, None, sequence=m_seq, dow_sequence=dow_seq)

    # Train evening model
    model_e = model_cls()
    if model_type in FEATURE_MODELS:
        model_e.fit(X_e, y_e.values)
    else:
        model_e.fit(None, None, sequence=e_seq, dow_sequence=dow_seq)

    return model_m, model_e, feature_cols_m


def _predict_single(model_m, model_e, model_type, X_pred_m, X_pred_e, last_m_digit, last_e_digit, current_dow=None):
    """Get probability distributions from a trained model pair."""
    if model_type in FEATURE_MODELS:
        m_probs = model_m.predict_proba(X_pred_m)
        e_probs = model_e.predict_proba(X_pred_e)
    else:
        m_probs = model_m.predict_proba(last_digit=last_m_digit, current_dow=current_dow)
        e_probs = model_e.predict_proba(last_digit=last_e_digit, current_dow=current_dow)
    return m_probs, e_probs


def _compute_metrics(predictions, actuals):
    """
    Compute evaluation metrics.

    Args:
        predictions: list of (m_probs, e_probs) per day
        actuals: list of (actual_m, actual_e) per day

    Returns dict with top1_m, top1_e, top3_m, top3_e, brier_m, brier_e, logloss_m, logloss_e
    """
    n = len(predictions)
    if n == 0:
        return None

    top1_m, top1_e = 0, 0
    top3_m, top3_e = 0, 0
    brier_m_sum, brier_e_sum = 0.0, 0.0
    logloss_m_sum, logloss_e_sum = 0.0, 0.0

    eps = 1e-15  # Prevent log(0)

    for (m_probs, e_probs), (act_m, act_e) in zip(predictions, actuals):
        act_m, act_e = int(act_m), int(act_e)

        # Top-1
        if np.argmax(m_probs) == act_m:
            top1_m += 1
        if np.argmax(e_probs) == act_e:
            top1_e += 1

        # Top-3
        if act_m in np.argsort(m_probs)[-3:]:
            top3_m += 1
        if act_e in np.argsort(e_probs)[-3:]:
            top3_e += 1

        # Brier Score (lower is better)
        m_onehot = np.zeros(10)
        m_onehot[act_m] = 1.0
        brier_m_sum += np.mean((m_probs - m_onehot) ** 2)

        e_onehot = np.zeros(10)
        e_onehot[act_e] = 1.0
        brier_e_sum += np.mean((e_probs - e_onehot) ** 2)

        # Log Loss (lower is better)
        logloss_m_sum += -np.log(max(m_probs[act_m], eps))
        logloss_e_sum += -np.log(max(e_probs[act_e], eps))

    return {
        'top1_m': top1_m / n, 'top1_e': top1_e / n,
        'top3_m': top3_m / n, 'top3_e': top3_e / n,
        'brier_m': brier_m_sum / n, 'brier_e': brier_e_sum / n,
        'logloss_m': logloss_m_sum / n, 'logloss_e': logloss_e_sum / n,
    }


def run_walk_forward(
    active_groups: list = None,
    num_periods: int = 5,
    pred_days: int = 7,
    verbose: bool = True,
):
    """
    Run full walk-forward validation.

    Returns:
        model_metrics: dict of model_id -> averaged metrics
        calibration_data: list of (predicted_top_prob, actual_hit) for calibration
    """
    if active_groups is None:
        active_groups = ALL_FEATURE_GROUPS.copy()

    df = load_raw_data()
    periods = _get_validation_periods(df, num_periods, pred_days)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD VALIDATION")
        print(f"  {len(periods)} periods × {pred_days} prediction days")
        print(f"  Active feature groups: {active_groups}")
        print(f"{'='*70}\n")

    # Collect metrics per model across all periods
    all_metrics = defaultdict(list)

    # Collect calibration data (predicted_top_prob, did_it_hit) across all models
    calibration_data_m = []
    calibration_data_e = []

    window_labels = ['1m', '2m', '3m', 'full']

    for period_idx, (train_end, pred_indices) in enumerate(periods):
        if verbose:
            print(f"  Period {period_idx+1}: Train up to {train_end}, predict {len(pred_indices)} days")

        # Split data
        train_mask = pd.to_datetime(df['Date']).dt.date <= train_end
        train_df = df[train_mask].copy()

        for wl in window_labels:
            for mt in MODEL_TYPES.keys():
                model_id = f"{wl}_{mt}"

                # Train
                model_m, model_e, feat_cols = _train_single_model(
                    mt, wl, train_df, active_groups
                )

                if model_m is None:
                    continue

                # Predict each validation day
                predictions = []
                actuals = []

                for pred_idx in pred_indices:
                    actual_m = int(df.iloc[pred_idx]['Morning_number'])
                    actual_e = int(df.iloc[pred_idx]['Evening_number'])

                    # Build features up to the day BEFORE the prediction day
                    context_df = df.iloc[:pred_idx].copy()
                    window_draws = get_window_size(wl)
                    context_sliced = slice_window(context_df, window_draws)

                    if len(context_sliced) < 5:
                        continue

                    feat_df_m, feat_df_e, _, _, _ = build_features(context_sliced, active_groups)
                    if len(feat_df_m) == 0:
                        continue

                    last_row_feats_m = feat_df_m.iloc[[-1]]
                    feat_only_m = [c for c in last_row_feats_m.columns if c != '_date']
                    X_pred_m = last_row_feats_m[feat_only_m].values
                    
                    last_row_feats_e = feat_df_e.iloc[[-1]]
                    feat_only_e = [c for c in last_row_feats_e.columns if c != '_date']
                    X_pred_e = last_row_feats_e[feat_only_e].values

                    last_m = int(context_df.iloc[-1]['Morning_number'])
                    last_e = int(context_df.iloc[-1]['Evening_number'])
                    
                    pred_date = pd.to_datetime(df.iloc[pred_idx]['Date'])
                    current_dow = pred_date.dayofweek

                    m_probs, e_probs = _predict_single(
                        model_m, model_e, mt, X_pred_m, X_pred_e, last_m, last_e, current_dow
                    )

                    predictions.append((m_probs, e_probs))
                    actuals.append((actual_m, actual_e))

                    # Collect calibration data
                    top_m_prob = np.max(m_probs)
                    top_e_prob = np.max(e_probs)
                    calibration_data_m.append((top_m_prob, int(np.argmax(m_probs) == actual_m)))
                    calibration_data_e.append((top_e_prob, int(np.argmax(e_probs) == actual_e)))

                if len(predictions) >= 3:
                    metrics = _compute_metrics(predictions, actuals)
                    if metrics:
                        all_metrics[model_id].append(metrics)

    # Average metrics across periods
    avg_metrics = {}
    for model_id, metric_list in all_metrics.items():
        avg = {}
        for key in metric_list[0].keys():
            avg[key] = np.mean([m[key] for m in metric_list])
        avg['n_periods'] = len(metric_list)
        avg_metrics[model_id] = avg

    if verbose:
        _print_metrics_table(avg_metrics)

    return avg_metrics, (calibration_data_m, calibration_data_e)


def _print_metrics_table(avg_metrics):
    """Pretty-print the metrics table."""
    print(f"\n{'='*90}")
    print(f"  {'Model':<15} {'Top1-M':>7} {'Top1-E':>7} {'Top3-M':>7} {'Top3-E':>7} "
          f"{'Brier-M':>8} {'Brier-E':>8} {'Periods':>7}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")

    for model_id in sorted(avg_metrics.keys()):
        m = avg_metrics[model_id]
        print(f"  {model_id:<15} {m['top1_m']:>6.1%} {m['top1_e']:>6.1%} "
              f"{m['top3_m']:>6.1%} {m['top3_e']:>6.1%} "
              f"{m['brier_m']:>8.4f} {m['brier_e']:>8.4f} {m['n_periods']:>7}")

    print(f"{'='*90}")
    print(f"  Random baseline: Top-1 = 10.0%, Top-3 = 30.0%, Brier = 0.1800")


def learn_weights(avg_metrics, temperature=0.02):
    """
    Compute model weights from Top-3 Accuracy via softmax.

    Higher Top-3 = better model = higher weight.
    Temperature controls sharpness:
      - Low temp (0.02) = winner-take-all
      - High temp (1.0) = nearly equal weights
    """
    model_ids = sorted(avg_metrics.keys())
    if not model_ids:
        return {}

    # Combined Top-3 = average of morning and evening
    top3_scores = []
    for mid in model_ids:
        m = avg_metrics[mid]
        combined_top3 = (m['top3_m'] + m['top3_e']) / 2.0
        top3_scores.append(combined_top3)

    top3_scores = np.array(top3_scores)

    # Softmax of POSITIVE Top-3 (higher Top-3 = higher score)
    raw_scores = top3_scores / temperature
    raw_scores -= raw_scores.max()  # Numerical stability
    exp_scores = np.exp(raw_scores)
    weights = exp_scores / exp_scores.sum()

    return {mid: float(w) for mid, w in zip(model_ids, weights)}


def prune_models(weights, cumulative_threshold=0.95):
    """
    Keep only models needed to reach cumulative_threshold weight.
    Returns dict of surviving model_id -> renormalized weight.
    """
    if not weights:
        return {}

    sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    survivors = {}
    cumulative = 0.0

    for mid, w in sorted_models:
        survivors[mid] = w
        cumulative += w
        if cumulative >= cumulative_threshold:
            break

    # Renormalize surviving weights
    total = sum(survivors.values())
    return {mid: w / total for mid, w in survivors.items()}


def build_calibration(calibration_data):
    """
    Build isotonic regression calibrator from validation data.

    Args:
        calibration_data: list of (predicted_top_prob, actual_hit_0_or_1)

    Returns:
        calibrator: fitted IsotonicRegression, or None if not enough data
        thresholds: dict of confidence level -> min predicted probability
    """
    if len(calibration_data) < 20:
        return None, {}

    preds = np.array([x[0] for x in calibration_data])
    hits = np.array([x[1] for x in calibration_data])

    # Fit isotonic regression: predicted_prob -> actual_hit_rate
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(preds, hits)

    # Determine evidence-based thresholds
    # Check calibrated hit rate at various predicted probability levels
    test_probs = np.linspace(0.08, 0.30, 50)
    calibrated = iso.predict(test_probs)

    thresholds = {}
    # Find the predicted prob level where calibrated hit rate crosses key levels
    # For a 10-class problem (random=10%), a hit rate of 16%+ is quite strong.
    for level_name, level_value in [('strong', 0.16), ('good', 0.14), ('marginal', 0.12)]:
        candidates = test_probs[calibrated >= level_value]
        if len(candidates) > 0:
            thresholds[level_name] = float(candidates[0])
        else:
            thresholds[level_name] = 1.0  # Unreachable = effectively disabled

    return iso, thresholds


def run_feature_ablation(base_groups=None, verbose=True):
    """
    Test each feature group by removing it and checking if performance drops.

    Returns list of surviving feature group names.
    """
    if base_groups is None:
        base_groups = ALL_FEATURE_GROUPS.copy()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  FEATURE ABLATION")
        print(f"{'='*70}\n")

    # Baseline: all groups
    if verbose:
        print("  Running baseline with ALL feature groups...")
    baseline_metrics, _ = run_walk_forward(base_groups, verbose=False)

    # Get baseline combined Brier (average across all models)
    baseline_brier = np.mean([
        (m['brier_m'] + m['brier_e']) / 2.0
        for m in baseline_metrics.values()
    ])
    if verbose:
        print(f"  Baseline avg Brier: {baseline_brier:.4f}\n")

    groups_to_remove = []

    for group in base_groups:
        reduced_groups = [g for g in base_groups if g != group]
        if verbose:
            print(f"  Testing WITHOUT '{group}'...", end=' ')

        reduced_metrics, _ = run_walk_forward(reduced_groups, verbose=False)

        if not reduced_metrics:
            if verbose:
                print("SKIP (no valid models)")
            continue

        reduced_brier = np.mean([
            (m['brier_m'] + m['brier_e']) / 2.0
            for m in reduced_metrics.values()
        ])

        diff = reduced_brier - baseline_brier
        if diff <= 0:
            # Removing this feature didn't hurt (or improved) — mark for removal
            if verbose:
                print(f"Brier={reduced_brier:.4f} (Δ={diff:+.4f}) → REMOVE ❌")
            groups_to_remove.append(group)
        else:
            if verbose:
                print(f"Brier={reduced_brier:.4f} (Δ={diff:+.4f}) → KEEP ✅")

    surviving_groups = [g for g in base_groups if g not in groups_to_remove]

    if verbose:
        print(f"\n  Surviving features: {surviving_groups}")
        print(f"  Removed features: {groups_to_remove}")

    return surviving_groups


def save_state(weights, surviving_groups, calibration_m, calibration_e,
               thresholds_m, thresholds_e, avg_metrics):
    """Save all learned state to disk."""
    _ensure_state_dir()

    state = {
        'weights': weights,
        'surviving_groups': surviving_groups,
        'thresholds_m': thresholds_m,
        'thresholds_e': thresholds_e,
        'timestamp': datetime.now().isoformat(),
        'model_metrics': {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in avg_metrics.items()},
    }

    path = os.path.join(STATE_DIR, 'state.json')
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)

    # Save calibrators separately (isotonic regression)
    if calibration_m is not None:
        cal_path = os.path.join(STATE_DIR, 'calibrator_m.json')
        with open(cal_path, 'w') as f:
            json.dump({
                'X_': calibration_m.X_.tolist() if hasattr(calibration_m, 'X_') else [],
                'y_': calibration_m.y_.tolist() if hasattr(calibration_m, 'y_') else [],
            }, f)

    if calibration_e is not None:
        cal_path = os.path.join(STATE_DIR, 'calibrator_e.json')
        with open(cal_path, 'w') as f:
            json.dump({
                'X_': calibration_e.X_.tolist() if hasattr(calibration_e, 'X_') else [],
                'y_': calibration_e.y_.tolist() if hasattr(calibration_e, 'y_') else [],
            }, f)

    print(f"\n  State saved to {path}")


def load_state():
    """Load learned state from disk."""
    path = os.path.join(STATE_DIR, 'state.json')
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        state = json.load(f)

    # Reconstruct calibrators
    calibrators = {}
    for target in ['m', 'e']:
        cal_path = os.path.join(STATE_DIR, f'calibrator_{target}.json')
        if os.path.exists(cal_path):
            with open(cal_path, 'r') as f:
                cal_data = json.load(f)
            if cal_data.get('X_') and cal_data.get('y_'):
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(np.array(cal_data['X_']), np.array(cal_data['y_']))
                calibrators[target] = iso

    state['calibrators'] = calibrators
    return state
