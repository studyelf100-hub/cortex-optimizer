"""
model.py — Train productivity prediction models and generate forecasts
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import prepare, get_feature_columns

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model.pkl')
META_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_meta.pkl')


def train(df: pd.DataFrame = None, verbose: bool = True):
    """
    Train all candidate models and return the best one.

    Returns:
        best_model: fitted sklearn model
        feature_cols: list of feature column names
        metrics: dict of evaluation metrics
    """
    if df is None:
        df = prepare(save=False)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df['productivity_score']

    if len(df) < 5:
        raise ValueError(f"Need at least 5 sessions to train. You have {len(df)}.")

    candidates = {
        'ridge': Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=6,
                                               min_samples_leaf=2, random_state=42),
        'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                     learning_rate=0.1, random_state=42),
    }

    cv_folds = min(5, len(df))
    results = {}

    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring='neg_mean_absolute_error')
        results[name] = {
            'model': model,
            'mae_cv': -scores.mean(),
            'mae_std': scores.std(),
        }
        if verbose:
            print(f"  {name:20s} MAE={-scores.mean():.3f} ±{scores.std():.3f}")

    best_name = min(results, key=lambda k: results[k]['mae_cv'])
    best = results[best_name]
    best['model'].fit(X, y)

    # Full-data metrics
    preds = best['model'].predict(X)
    metrics = {
        'best_model': best_name,
        'mae': mean_absolute_error(y, preds),
        'r2': r2_score(y, preds),
        'mae_cv': best['mae_cv'],
        'n_sessions': len(df),
        'feature_cols': feature_cols,
    }

    if verbose:
        print(f"\n✓ Best model: {best_name}")
        print(f"  MAE (train): {metrics['mae']:.3f}")
        print(f"  R²  (train): {metrics['r2']:.3f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best['model'], f)
    with open(META_PATH, 'wb') as f:
        pickle.dump(metrics, f)

    return best['model'], feature_cols, metrics


def load_model():
    """Load saved model and metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No trained model found. Run train() first.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)
    return model, meta


def predict_grid(model, feature_cols: list, subjects: list,
                 hours: list = None, duration: int = 90) -> pd.DataFrame:
    """
    Predict productivity for all (subject × hour) combinations.

    Args:
        model: trained model
        feature_cols: list of feature column names
        subjects: list of subject names
        hours: list of hours to evaluate (default: 6–23)
        duration: session duration in minutes

    Returns:
        DataFrame with columns: subject, hour, predicted_score
    """
    if hours is None:
        hours = list(range(6, 24))

    rows = []
    for subject in subjects:
        for hour in hours:
            row = _build_feature_row(hour, subject, duration, feature_cols)
            rows.append({'subject': subject, 'hour': hour, 'features': row})

    X = pd.DataFrame([r['features'] for r in rows]).fillna(0)
    preds = model.predict(X)
    preds = np.clip(preds, 1, 10)

    result = pd.DataFrame([
        {'subject': r['subject'], 'hour': r['hour'], 'predicted_score': round(preds[i], 2)}
        for i, r in enumerate(rows)
    ])
    return result


def _build_feature_row(hour: int, subject: str, duration: int, feature_cols: list) -> dict:
    """Build a single feature row matching the training schema."""
    row = {col: 0 for col in feature_cols}

    row['time_sin'] = np.sin(2 * np.pi * hour / 24)
    row['time_cos'] = np.cos(2 * np.pi * hour / 24)
    row['duration_norm'] = min(duration, 240) / 240

    subj_col = f'subj_{subject}'
    if subj_col in row:
        row[subj_col] = 1

    # Day of week: assume Tuesday (weekday)
    if 'day_of_week' in row:
        row['day_of_week'] = 1
    if 'is_weekend' in row:
        row['is_weekend'] = 0

    return row


if __name__ == '__main__':
    print("Training model...\n")
    model, feature_cols, metrics = train(verbose=True)
    print(f"\nModel saved to {MODEL_PATH}")
