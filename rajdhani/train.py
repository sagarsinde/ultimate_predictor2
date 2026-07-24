"""
Train production models for RAJDHANI using state.json intelligence.
Reads surviving feature groups and model weights from the backtest.
"""

import sys
import os
import json
import pandas as pd

# Add the market directory to sys.path so we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import load_raw_data, slice_window, get_window_size, build_features, ALL_FEATURE_GROUPS
from src.models import MODEL_TYPES, FEATURE_MODELS, SEQUENCE_MODELS

def train_full_data():
    print(f"\n--- Training Production Models for RAJDHANI ---")
    
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(dir_path, exist_ok=True)
    state_path = os.path.join(dir_path, 'state.json')
    
    active_groups = ALL_FEATURE_GROUPS
    target_models = []
    
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            if 'surviving_groups' in state:
                active_groups = state['surviving_groups']
                print(f"Loaded {len(active_groups)} surviving feature groups from state.json.")
            if 'weights' in state:
                target_models = list(state['weights'].keys())
                print(f"Will only train the {len(target_models)} models that survived pruning.")
    else:
        print("WARNING: state.json not found! You must run run_backtest.py first to find optimal features.")
        return

    df = load_raw_data()
    print(f"Loaded {len(df)} draws (Full History).")
    
    # Pre-build global features using ONLY the surviving groups!
    full_feat_m, full_feat_e, full_y_m, full_y_e, _ = build_features(df, active_groups)

    print("========================================")
    print("  RAJDHANI: PRODUCTION TRAINING")
    print("========================================")
    
    window_labels = ['1m', '2m', '3m', 'full']
    
    for wl in window_labels:
        window_draws = get_window_size(wl)
        
        for name, ModelClass in MODEL_TYPES.items():
            if name == 'ag': continue
            
            model_id = f"{wl}_{name}"
            if target_models and model_id not in target_models:
                continue # Skip models that didn't survive pruning
                
            print(f"  -> Training {model_id}...")
            
            # Slice features for the window
            if window_draws:
                X_m_df = full_feat_m.iloc[-window_draws:]
                X_e_df = full_feat_e.iloc[-window_draws:]
                y_m = full_y_m.iloc[-window_draws:]
                y_e = full_y_e.iloc[-window_draws:]
            else:
                X_m_df = full_feat_m
                X_e_df = full_feat_e
                y_m = full_y_m
                y_e = full_y_e
                
            if len(X_m_df) < 5:
                continue
                
            X_m = X_m_df.drop(columns=['_date'], errors='ignore').values
            X_e = X_e_df.drop(columns=['_date'], errors='ignore').values
            seq_m = y_m.values
            seq_e = y_e.values
            
            model_m = ModelClass()
            model_e = ModelClass()
            
            if name in FEATURE_MODELS:
                model_m.fit(X_m, seq_m)
                model_e.fit(X_e, seq_e)
            elif name in SEQUENCE_MODELS:
                model_m.fit(None, None, sequence=seq_m)
                model_e.fit(None, None, sequence=seq_e)
                
            model_m.save_models(dir_path, f"{model_id}_morning")
            model_e.save_models(dir_path, f"{model_id}_evening")
            
    print(f"Success! Production models saved to {dir_path}/")

if __name__ == '__main__':
    train_full_data()
