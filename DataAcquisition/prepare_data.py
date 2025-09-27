# data_preparation.py

"""
This script processes the raw JSON output from RTR NetTest API into
structured datasets suitable for machine learning training, validation,
and testing.

Steps:
1. Load and normalize the JSON data.
2. Spatial structuring (latitude/longitude handling).
3. Feature normalization (scaling).
4. Split into training, validation, and test sets.
5. Save processed splits to CSV files.
"""

import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Constants
RAW_JSON_PATH = os.path.join("outputs", "rtr_data.json")
PROCESSED_DIR = "processed_data"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # percentage of full dataset

# Desired features to extract; will keep only those actually present
DESIRED_FEATURES = [
    "lat", "long", "loc_accuracy",
    "download_kbit", "upload_kbit", "ping_ms",
    "signal_strength", "lte_rsrp","model", "platform", "provider_name"
]


def load_json(path):
    """Load the JSON file and return list of records."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', [])


def prepare_dataframe(records):
    """Convert raw records to pandas DataFrame and filter to available features."""
    df = pd.json_normalize(records)

    # Determine which desired features are present
    available = [col for col in DESIRED_FEATURES if col in df.columns]
    missing = set(DESIRED_FEATURES) - set(available)
    if missing:
        print(f"Warning: The following desired features are missing and will be skipped: {sorted(missing)}")

    df = df[available]

    device_cond = (
            df['platform'].str.contains("iOS", case=False, na=False) |
            df['platform'].str.contains("Android", case=False, na=False) |
            df['model'].str.contains("iPhone", case=False, na=False)
    )

    provider_cond = df['provider_name'].str.contains("A1 TA", case=False, na=False)

    condition = device_cond & provider_cond

    df = df[condition]

    df = df.dropna(how='all', subset=available)
    return df


def spatial_structuring(df):
    """Optionally transform lat/long into grid cells or keep raw."""
    # Here we keep raw lat/long; can add clustering or binning later
    return df


def normalize_features(df):
    """Apply standard scaling to numerical features only."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    df_scaled_numeric = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)

    # Concatenate back with non-numeric columns (unchanged)
    df_final = pd.concat([df_scaled_numeric, df[non_numeric_cols]], axis=1)

    return df_final, scaler



def split_data(df):
    """Split into train, validation, and test sets."""
    # First split off test set
    train_val, test = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    # Then split train+val into train and validation
    val_relative = VALIDATION_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=42)
    return train, val, test


def save_splits(train, val, test):
    """Save each split to CSV files."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "validation.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)


def main():
    # Load and prepare
    records = load_json(RAW_JSON_PATH)
    df = prepare_dataframe(records)
    df = spatial_structuring(df)

    # Normalize
    df_scaled, scaler = normalize_features(df)

    # Split
    train, val, test = split_data(df_scaled)

    # Save outputs
    save_splits(train, val, test)
    print("Data preparation complete.")
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")


if __name__ == "__main__":
    main()
