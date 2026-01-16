import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# data_loader is only needed if processed file doesn't exist
from data_loader import load_all_data
from feature_eng import DataProcessor
from model import RULPredictor
from sklearn.model_selection import train_test_split

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WINDOW_SIZE = 30
TEST_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    print(f"🚀 Training started on device: {DEVICE}")

    processor = DataProcessor(window_size=WINDOW_SIZE)
    processed_path = "../data/processed/final_features.csv"

    # --- SMART DATA LOADING ---
    if os.path.exists(processed_path):
        print(" Found processed data. Loading directly from CSV...")
        # Load directly (pass None to raw_df as it won't be used)
        feat_df = processor.extract_features(None)
    else:
        print(" Processed data not found. Reading RAW .mat files (This may take time)...")
        raw_df = load_all_data("../data/raw")
        if raw_df is None:
            print(" Error: No data found.")
            return
        # This will save the CSV for next time
        feat_df = processor.extract_features(raw_df)

    # Prepare Tensors
    X, y, scaler = processor.prepare_lstm_data(feat_df)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=TEST_SIZE, random_state=42)

    print(f"Train Set: {X_train.shape}, Validation Set: {X_val.shape}")

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = RULPredictor(input_dim=4, hidden_dim=64, num_layers=2).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            predictions, _ = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                predictions, _ = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists("../models"):
                os.makedirs("../models")
            torch.save(model.state_dict(), "../models/best_model.pth")
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ✅ Model Saved")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("\n Training Complete!")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Process: Loss Reduction')
    plt.legend()
    plt.grid(True)

    if not os.path.exists("../figures"):
        os.makedirs("../figures")
    plt.savefig("../figures/loss_curve.png")
    print("📈 Loss curve saved to figures/loss_curve.png")


if __name__ == "__main__":
    train_model()