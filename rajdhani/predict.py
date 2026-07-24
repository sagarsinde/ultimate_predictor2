"""
Prediction Script for Rajdhani
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import timedelta

# Add the rajdhani directory to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import load_raw_data, build_prediction_features, ALL_FEATURE_GROUPS, PLAYING_DAYS_PER_WEEK
from src.models import MODEL_TYPES

def get_next_playing_date(last_date):
    next_date = last_date + timedelta(days=1)
    if PLAYING_DAYS_PER_WEEK == 6:
        while next_date.weekday() == 6: next_date += timedelta(days=1)
    elif PLAYING_DAYS_PER_WEEK == 5:
        while next_date.weekday() >= 5: next_date += timedelta(days=1)
    return next_date

def format_prediction(probs, thresholds=None):
    top_4_idx = np.argsort(probs)[-4:][::-1]
    top_prob = probs[top_4_idx[0]]
    
    confidence = ""
    if thresholds:
        if top_prob >= thresholds.get('strong', 1.0): confidence = " [STRONG]"
        elif top_prob >= thresholds.get('good', 1.0): confidence = " [GOOD]"
        elif top_prob >= thresholds.get('marginal', 1.0): confidence = " [MARGINAL]"
        else: confidence = " [WEAK]"

    output = []
    for idx in top_4_idx:
        pct = probs[idx] * 100
        output.append(f"{idx} ({pct:.1f}%)")
    return " | ".join(output) + confidence

def run_prediction():
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(dir_path):
        print(f"ERROR: Cannot find trained models at {dir_path}")
        sys.exit(1)
        
    print(f"\n{'#'*70}")
    print(f"  PREDICTING TOMORROW: RAJDHANI")
    print(f"{'#'*70}")

    df = load_raw_data()
    last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
    pred_date = get_next_playing_date(last_date)
    
    print(f"\nLast Draw: {last_date}  -->  Predicting For: {pred_date.strftime('%Y-%m-%d (%A)')}")
    
    weights = None
    thresholds_m, thresholds_e = None, None
    surviving_groups = ALL_FEATURE_GROUPS
    
    state_path = os.path.join(dir_path, "state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
        weights = state.get('weights', {})
        thresholds_m = state.get('thresholds_m', {})
        thresholds_e = state.get('thresholds_e', {})
        surviving_groups = state.get('surviving_groups', ALL_FEATURE_GROUPS)
        print("Loaded intelligent weights & calibrations from state.json")

    print("Building features for tomorrow...")
    X_m_last, X_e_last, _ = build_prediction_features(df, surviving_groups)
    
    seq_m = df['Morning_number'].dropna().astype(int).values
    seq_e = df['Evening_number'].dropna().astype(int).values
    current_dow = pred_date.weekday()

    final_probs_m = np.zeros(10)
    final_probs_e = np.zeros(10)
    
    loaded_m = []
    loaded_e = []

    print("\n--- Model Predictions ---")
    
    active_keys = set(weights.keys()) if weights else None

    for name, ModelClass in MODEL_TYPES.items():
        if name == 'ag': continue
        
        for window in ['1m', '2m', '3m', 'full']:
            model_id = f"{window}_{name}"
            
            # Skip models not in the pruned set
            if active_keys is not None and model_id not in active_keys:
                continue
            
            w = weights[model_id] if weights and model_id in weights else 1.0
            
            # Morning
            model_m = ModelClass()
            try:
                model_m.load_models(dir_path, f"{model_id}_morning")
                probs_m = model_m.predict_proba(X_m_last, last_digits=seq_m, current_dow=current_dow)
                final_probs_m += probs_m * w
                loaded_m.append(model_id)
            except Exception:
                pass
            
            # Evening
            model_e = ModelClass()
            try:
                model_e.load_models(dir_path, f"{model_id}_evening")
                probs_e = model_e.predict_proba(X_e_last, last_digits=seq_e, current_dow=current_dow)
                final_probs_e += probs_e * w
                loaded_e.append(model_id)
            except Exception:
                pass

    if not loaded_m and not loaded_e:
        print("Error: No models were successfully loaded! Check if the models folder has files.")
        return

    # Normalize if not using state weights
    if not weights and sum(final_probs_m) > 0: final_probs_m /= sum(final_probs_m)
    if not weights and sum(final_probs_e) > 0: final_probs_e /= sum(final_probs_e)


    print(f"Loaded {len(loaded_m)} Morning Models")
    print(f"Loaded {len(loaded_e)} Evening Models")
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE MORNING  : {format_prediction(final_probs_m, thresholds_m)}")
    print(f"ENSEMBLE EVENING  : {format_prediction(final_probs_e, thresholds_e)}")
    
    # Calculate Jodi outer product
    jodi_probs = np.outer(final_probs_m, final_probs_e)
    flat_jodi_probs = jodi_probs.flatten()
    jodi_ranking = np.argsort(flat_jodi_probs)[::-1]
    
    top4_jodis = []
    for rank in jodi_ranking[:4]:
        m_digit = rank // 10
        e_digit = rank % 10
        prob_pct = flat_jodi_probs[rank] * 100
        top4_jodis.append(f"{m_digit}{e_digit} ({prob_pct:.1f}%)")
        
    print(f"ENSEMBLE JODI     : {' | '.join(top4_jodis)}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    run_prediction()
