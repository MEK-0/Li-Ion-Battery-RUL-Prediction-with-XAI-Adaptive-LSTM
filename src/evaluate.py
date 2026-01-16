import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import RULPredictor
from feature_eng import DataProcessor
from data_loader import load_all_data

# --- CONFIGURATION ---
MODEL_PATH = "../models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30


def plot_results(y_true, y_pred, battery_id):
    """
    Plots Actual vs Predicted RUL.
    With Adaptive Input, the prediction line starts from Cycle 0.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual RUL (Ground Truth)', color='black', linestyle='--')
    plt.plot(y_pred, label='Predicted RUL (Adaptive-LSTM)', color='blue', linewidth=2)
    plt.title(f'RUL Prediction: {battery_id} (Adaptive Input Mode)')
    plt.xlabel('Time Steps (Cycles)')
    plt.ylabel('Remaining Useful Life (Cycles)')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("../figures"):
        os.makedirs("../figures")
    save_path = f"../figures/RUL_prediction_Final_{battery_id}.png"
    plt.savefig(save_path)
    print(f" Prediction plot saved to {save_path}")


def plot_attention(weights, battery_id):
    """
    Visualizes the Attention Weights heatmap.
    """
    # Show last 50 steps for better visibility
    subset_size = 50
    if len(weights) > subset_size:
        attn_map = weights[-subset_size:].squeeze().T
    else:
        attn_map = weights.squeeze().T

    plt.figure(figsize=(12, 6))
    sns.heatmap(attn_map, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
    plt.title(f'XAI: Attention Mechanism Map ({battery_id})')
    plt.xlabel('Prediction Step (Recent Cycles)')
    plt.ylabel('Window Lag')

    save_path = f"../figures/XAI_Attention_Final_{battery_id}.png"
    plt.savefig(save_path)
    print(f" XAI Heatmap saved to {save_path}")


def prepare_adaptive_input(bat_data, processor):
    """
    ADAPTIVE WINDOWING LOGIC:
    Handles the 'Cold Start' problem where cycle count < WINDOW_SIZE.
    Instead of skipping early cycles, we use 'Edge Padding'.
    """
    feature_values = bat_data[processor.feature_columns].values
    target_values = bat_data[processor.target_column].values

    X_sequences = []
    y_targets = []

    total_cycles = len(bat_data)

    for i in range(total_cycles):
        # We need data from (i - window_size) to (i)

        if i < WINDOW_SIZE:
            # --- ADAPTIVE MODE (PADDING) ---
            # Not enough history. Take available data from 0 to i.
            available_data = feature_values[0: i + 1]

            # Calculate missing length
            missing_len = WINDOW_SIZE - len(available_data)

            # Edge Padding: Repeat the first available row to fill the gap.
            # This simulates a steady state before the experiment started.
            first_row = available_data[0].reshape(1, -1)
            padding = np.repeat(first_row, missing_len, axis=0)

            # Stack Padding + Available Data
            window = np.vstack((padding, available_data))

        else:
            # --- NORMAL MODE ---
            # Enough history available. Standard sliding window.
            window = feature_values[i - WINDOW_SIZE + 1: i + 1]

        # Target: Current RUL
        target = target_values[i]

        X_sequences.append(window)
        y_targets.append(target)

    return np.array(X_sequences), np.array(y_targets)


def evaluate_model():
    print(f" Evaluating model on {DEVICE}...")
    print(" Mode: ADAPTIVE INPUT (Padding Enabled)")

    # 1. Load Raw Data
    raw_df = load_all_data("../data/raw")
    if raw_df is None: return

    processor = DataProcessor(window_size=WINDOW_SIZE)
    feat_df = processor.extract_features(raw_df)

    # Scale Data (Using a new fit for demonstration)
    # Ideally, load the scaler saved during training.
    feat_df_scaled = feat_df.copy()
    feat_df_scaled[processor.feature_columns] = processor.scaler.fit_transform(feat_df[processor.feature_columns])

    # 2. Load Model
    model = RULPredictor(input_dim=4, hidden_dim=64, num_layers=2).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(" Pre-trained model loaded successfully.")
    else:
        print(" Model file not found! Please run train.py first.")
        return

    model.eval()

    # 3. Test Specific Batteries
    test_batteries = ['B0005', 'B0032', 'B0044']

    for bat_id in test_batteries:
        if bat_id not in feat_df_scaled['battery_id'].values:
            print(f" Battery {bat_id} not found. Skipping.")
            continue

        print(f"\n--- Testing on Battery: {bat_id} ---")
        bat_data = feat_df_scaled[feat_df_scaled['battery_id'] == bat_id]

        # --- APPLY ADAPTIVE INPUT ---
        X, y = prepare_adaptive_input(bat_data, processor)

        X_tensor = torch.FloatTensor(X).to(DEVICE)

        # Predict
        with torch.no_grad():
            preds, attention_weights = model(X_tensor)

        preds = preds.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()

        # Calculate RMSE
        mse = np.mean((preds - y.reshape(-1, 1)) ** 2)
        rmse = np.sqrt(mse)
        print(f" RMSE for {bat_id}: {rmse:.4f} cycles")

        # Plot Results
        plot_results(y, preds, bat_id)
        plot_attention(attention_weights, bat_id)


if __name__ == "__main__":
    evaluate_model()