"""
scheduler.py — Optimal study schedule recommendation engine
"""

import numpy as np
import pandas as pd
from model import predict_grid


def generate_schedule(
    model,
    feature_cols: list,
    subjects: list,
    hours_available: list = None,
    sessions_per_day: int = 3,
    default_duration: int = 90,
    gap_minutes: int = 15,
) -> dict:
    """
    Generate an optimal daily study schedule.

    Strategy:
    1. Predict scores for all (subject, hour) pairs.
    2. Greedily pick top sessions, enforcing time gaps.

    Returns:
        dict with 'schedule', 'insights', 'warnings'
    """
    if hours_available is None:
        hours_available = list(range(6, 23))

    grid = predict_grid(model, feature_cols, subjects,
                        hours=hours_available, duration=default_duration)

    # Sort by score descending
    grid = grid.sort_values('predicted_score', ascending=False).reset_index(drop=True)

    schedule = []
    used_hours = set()

    for _, row in grid.iterrows():
        if len(schedule) >= sessions_per_day:
            break

        hour = row['hour']
        # Avoid overlap: require `gap_minutes` / 60 hours gap
        gap_h = np.ceil((default_duration + gap_minutes) / 60)

        conflict = any(abs(hour - h) < gap_h for h in used_hours)
        if conflict:
            continue

        schedule.append({
            'start': f'{int(hour):02d}:00',
            'end': f'{int(hour + default_duration // 60):02d}:{default_duration % 60:02d}',
            'subject': row['subject'],
            'predicted_score': row['predicted_score'],
        })
        used_hours.add(hour)

    schedule.sort(key=lambda x: x['start'])

    insights = _extract_insights(grid, subjects)
    warnings = _generate_warnings(grid, default_duration)

    return {
        'schedule': schedule,
        'insights': insights,
        'warnings': warnings,
        'grid': grid,
    }


def _extract_insights(grid: pd.DataFrame, subjects: list) -> list:
    insights = []

    # Peak hour overall
    top = grid.iloc[0]
    insights.append(
        f"Peak performance window: {int(top['hour'])}:00 "
        f"(predicted score {top['predicted_score']:.1f}/10)"
    )

    # Best subject
    subj_means = grid.groupby('subject')['predicted_score'].mean().sort_values(ascending=False)
    if len(subj_means) >= 2:
        diff = subj_means.iloc[0] - subj_means.iloc[1]
        insights.append(
            f"{subj_means.index[0]} outperforms other subjects "
            f"by +{diff:.1f} points on average"
        )

    # Low-performance period
    hour_means = grid.groupby('hour')['predicted_score'].mean()
    worst_hour = hour_means.idxmin()
    insights.append(
        f"Avoid studying at {int(worst_hour)}:00 "
        f"(lowest predicted score: {hour_means.min():.1f})"
    )

    return insights


def _generate_warnings(grid: pd.DataFrame, duration: int) -> list:
    warnings = []
    if duration > 120:
        warnings.append(
            "Sessions > 120 min often show diminishing returns. "
            "Consider splitting into two focused blocks."
        )
    if grid['predicted_score'].max() - grid['predicted_score'].min() > 2.5:
        warnings.append(
            "High variance detected across time slots — "
            "timing matters significantly for your performance."
        )
    return warnings


def print_schedule(result: dict):
    print("\n" + "═" * 52)
    print("  RECOMMENDED STUDY SCHEDULE")
    print("═" * 52)

    for s in result['schedule']:
        bar = '█' * int(s['predicted_score'])
        print(f"  {s['start']} → {s['end']}  {s['subject']}")
        print(f"    Score: {s['predicted_score']:.1f}/10  {bar}")
        print()

    print("INSIGHTS")
    print("─" * 52)
    for ins in result['insights']:
        print(f"  • {ins}")

    if result['warnings']:
        print("\nWARNINGS")
        print("─" * 52)
        for w in result['warnings']:
            print(f"  ⚠ {w}")

    print("═" * 52 + "\n")


if __name__ == '__main__':
    from model import load_model
    from preprocess import prepare, get_feature_columns

    model, meta = load_model()
    df = prepare(save=False)
    subjects = df['subject'].unique().tolist()
    feature_cols = meta['feature_cols']

    result = generate_schedule(model, feature_cols, subjects)
    print_schedule(result)
