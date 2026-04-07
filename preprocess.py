"""
preprocess.py — Feature engineering for study session data
"""

import numpy as np
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sessions.csv')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')


def load_raw() -> pd.DataFrame:
    """Load raw session data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No data found at {DATA_PATH}. Log some sessions first.")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    return df


def encode_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour of day as sine/cosine to preserve cyclical structure."""
    df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
    return df


def encode_subject(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode subject column."""
    dummies = pd.get_dummies(df['subject'], prefix='subj')
    df = pd.concat([df, dummies], axis=1)
    return df


def normalize_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize duration to 0–1 range (cap at 240 minutes)."""
    df['duration_norm'] = (df['duration_minutes'].clip(0, 240) / 240).round(4)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = encode_time_cyclical(df)
    df = encode_subject(df)
    df = normalize_duration(df)

    # Day of week (0=Monday)
    if 'date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (excluding target and metadata)."""
    exclude = {'date', 'subject', 'time_of_day', 'productivity_score', 'duration_minutes'}
    return [c for c in df.columns if c not in exclude]


def prepare(save: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline: load → engineer → save."""
    df = load_raw()
    df = engineer_features(df)

    if save:
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"✓ Processed {len(df)} sessions → {PROCESSED_PATH}")

    return df


if __name__ == '__main__':
    df = prepare()
    print(df.head())
    print(f"\nFeatures: {get_feature_columns(df)}")
