import scipy.io
import numpy as np
import pandas as pd
import os
import glob


def load_battery_data(mat_file_path):
    """
    Loads a single NASA PCoE .mat file and extracts discharge cycle data.

    Args:
        mat_file_path (str): Path to the .mat file.

    Returns:
        pd.DataFrame: DataFrame containing cycle-level data or None if error occurs.
    """
    if not os.path.exists(mat_file_path):
        return None

    # Extract filename (e.g., 'B0005') to use as ID
    battery_id = os.path.basename(mat_file_path).split('.')[0]

    try:
        data = scipy.io.loadmat(mat_file_path)
        # Access the top-level structure (variable name matches filename)
        if battery_id not in data:
            # Sometimes variable name might differ, handle edge cases if necessary
            print(f"Warning: Variable {battery_id} not found in mat file.")
            return None

        col = data[battery_id]
        col = col[0][0][0][0]
        size = col.shape[0]

        data_list = []

        for i in range(size):
            k = list(col[i])

            # k[0] -> type (charge, discharge, impedance)
            operation_type = str(k[0][0])

            # We only care about discharge cycles for RUL prediction
            if operation_type == 'discharge':
                # k[1] -> ambient_temperature (from dataset metadata)
                ambient_temp = int(k[1][0])

                # k[2] -> date_time (string)
                date_time = str(k[2][0])

                # k[3] -> data structure (Measurements)
                measurements = k[3][0][0]

                # NASA Structure for Discharge:
                # measurements[0] -> Voltage_measured
                # measurements[1] -> Current_measured
                # measurements[2] -> Temperature_measured
                # measurements[5] -> Time (seconds)
                # measurements[6] -> Capacity (Ahr) - TARGET LABEL

                # Check if capacity data exists (some old datasets might miss it, but PCoE usually has it)
                if len(measurements[6]) > 0:
                    capacity = measurements[6][0][0]

                    data_list.append({
                        'battery_id': battery_id,
                        'cycle': i + 1,  # Original index
                        'ambient_temp': ambient_temp,
                        'datetime': date_time,
                        'capacity': capacity,
                        'voltage_measured': measurements[0][0],  # Array
                        'current_measured': measurements[1][0],  # Array
                        'temperature_measured': measurements[2][0],  # Array
                        'time_sec': measurements[5][0]  # Array
                    })

        df = pd.DataFrame(data_list)

        # Re-index cycles to be sequential (1, 2, 3...) since we filtered only discharge
        if not df.empty:
            df = df.reset_index(drop=True)
            df['cycle_count'] = df.index + 1
            return df
        else:
            return None

    except Exception as e:
        print(f"Error processing {battery_id}: {e}")
        return None


def load_all_data(base_path="../data/raw"):
    """
    Traverses the base directory to find all .mat files in subfolders (24C, 43C, 04C).
    Combines them into a single DataFrame.
    """
    # Find all .mat files recursively
    search_path = os.path.join(base_path, "**", "*.mat")
    all_files = glob.glob(search_path, recursive=True)

    if not all_files:
        print(f"No .mat files found in {base_path}")
        return None

    print(f"Found {len(all_files)} battery files. Loading...")

    all_dfs = []
    for file_path in all_files:
        print(f" -> Processing: {os.path.basename(file_path)}", end=" ")
        df = load_battery_data(file_path)
        if df is not None:
            all_dfs.append(df)
            print("[OK]")
        else:
            print("[FAILED]")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nData Loading Complete!")
        print(f"Total Cycles: {len(final_df)}")
        print(f"Unique Batteries: {final_df['battery_id'].unique()}")
        print(f"Ambient Temperatures: {final_df['ambient_temp'].unique()}")
        return final_df
    else:
        return None


if __name__ == "__main__":
    # Test Block
    # Ensure this runs correctly from the 'src' directory
    print("Testing data loader...")
    df_test = load_all_data(base_path="../data/raw")

    if df_test is not None:
        print("\nSample Data:")
        print(df_test[['battery_id', 'cycle_count', 'ambient_temp', 'capacity']].head())