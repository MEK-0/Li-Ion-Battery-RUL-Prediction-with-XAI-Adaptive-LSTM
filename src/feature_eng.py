import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os


class DataProcessor:
    """
    Handles feature extraction, normalization, and saving processed data.
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = ['discharge_time', 'min_voltage', 'max_temp', 'ambient_temp']
        self.target_column = 'RUL'
        # Processed data path
        self.save_path = "../data/processed/final_features.csv"

    def extract_features(self, df_raw):
        """
        Extracts features and SAVES them to data/processed/.
        """
        # Check if processed file already exists to save time
        if os.path.exists(self.save_path):
            print(f"⚡ Processed data found at {self.save_path}. Loading from disk...")
            df_features = pd.read_csv(self.save_path)
            return df_features

        print(f"⚙️ Starting Feature Extraction on {len(df_raw)} cycles...")
        processed_data = []
        batteries = df_raw['battery_id'].unique()

        for battery in tqdm(batteries, desc="Processing Batteries"):
            battery_df = df_raw[df_raw['battery_id'] == battery].sort_values(by='cycle_count')
            total_cycles = battery_df['cycle_count'].max()

            for index, row in battery_df.iterrows():
                time_vector = np.array(row['time_sec'])
                voltage_vector = np.array(row['voltage_measured'])
                temp_vector = np.array(row['temperature_measured'])

                # Feature Engineering
                if len(time_vector) > 0:
                    discharge_time = time_vector[-1] - time_vector[0]
                else:
                    discharge_time = 0

                min_vol = np.min(voltage_vector) if len(voltage_vector) > 0 else 0
                max_temp = np.max(temp_vector) if len(temp_vector) > 0 else 0
                amb_temp = row['ambient_temp']
                rul = total_cycles - row['cycle_count']

                processed_data.append({
                    'battery_id': battery,
                    'cycle': row['cycle_count'],
                    'discharge_time': discharge_time,
                    'min_voltage': min_vol,
                    'max_temp': max_temp,
                    'ambient_temp': amb_temp,
                    'RUL': rul
                })

        df_features = pd.DataFrame(processed_data)

        # --- SAVE TO PROCESSED FOLDER ---
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df_features.to_csv(self.save_path, index=False)
        print(f" Data saved to {self.save_path}")
        print(f"Feature Extraction Completed. Shape: {df_features.shape}")

        return df_features

    def prepare_lstm_data(self, df_features):
        """
        Converts summary features into Sliding Windows for LSTM.
        """
        print("Creating LSTM Sequences...")

        df_scaled = df_features.copy()
        df_scaled[self.feature_columns] = self.scaler.fit_transform(df_features[self.feature_columns])

        X_sequences = []
        y_targets = []

        batteries = df_scaled['battery_id'].unique()

        for battery in batteries:
            bat_data = df_scaled[df_scaled['battery_id'] == battery].sort_values(by='cycle')
            feature_values = bat_data[self.feature_columns].values
            target_values = bat_data[self.target_column].values

            num_cycles = len(bat_data)
            if num_cycles < self.window_size:
                continue

            for i in range(num_cycles - self.window_size):
                window = feature_values[i: i + self.window_size]
                target = target_values[i + self.window_size]
                X_sequences.append(window)
                y_targets.append(target)

        X = np.array(X_sequences)
        y = np.array(y_targets)

        print(f" LSTM Data Prepared! X: {X.shape}, y: {y.shape}")
        return X, y, self.scaler


if __name__ == "__main__":
    # Test Block
    from data_loader import load_all_data

    # If file exists, it will load from CSV. If not, it will load raw data first.
    if os.path.exists("../data/processed/final_features.csv"):
        print("Loading directly from Processed folder...")
        processor = DataProcessor(window_size=30)
        # We pass None because we expect it to load from CSV
        feat_df = processor.extract_features(None)
    else:
        print("Loading Raw Data first...")
        raw_df = load_all_data("../data/raw")
        processor = DataProcessor(window_size=30)
        feat_df = processor.extract_features(raw_df)

    print(feat_df.head())