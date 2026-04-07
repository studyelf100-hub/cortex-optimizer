"""
seed_data.py — Generate realistic demo sessions for testing
"""

import csv
import os
import random
import numpy as np
from datetime import datetime, timedelta

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sessions.csv')
FIELDNAMES = ['date', 'time_of_day', 'subject', 'duration_minutes', 'productivity_score']


def generate_sessions(n: int = 80, seed: int = 42) -> list:
    """
    Generate n realistic study sessions with plausible patterns:
    - Peak performance 17:00–20:00
    - Diminishing returns after 100 min
    - Subject-specific variance
    - Weekend slight dip
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    subjects = ['Mathematics', 'Physics', 'Programming', 'Chemistry', 'Literature']

    # Subject base scores
    subject_base = {
        'Mathematics': 6.8,
        'Physics': 6.2,
        'Programming': 7.1,
        'Chemistry': 5.9,
        'Literature': 6.5,
    }

    sessions = []
    base_date = datetime(2024, 9, 1)

    for i in range(n):
        date = base_date + timedelta(days=rng.randint(0, 120))
        hour = rng.choices(
            range(6, 24),
            weights=[1, 1, 2, 2, 3, 5, 6, 8, 8, 10, 8, 6, 5, 4, 3, 3, 3, 2],
            k=1
        )[0]
        subject = rng.choice(subjects)
        duration = rng.choice([30, 45, 60, 75, 90, 90, 105, 120, 135, 150, 180])

        # Base score from time-of-day curve
        time_score = 4.5 + 3.5 * np.sin(np.pi * (hour - 6) / 14) * (hour >= 6 and hour <= 21)

        # Subject modifier
        subj_mod = subject_base[subject] - 6.5

        # Duration penalty
        duration_penalty = max(0, (duration - 100) * 0.015)

        # Weekend dip
        weekday = date.weekday()
        weekend_penalty = 0.4 if weekday >= 5 else 0.0

        score = time_score + subj_mod - duration_penalty - weekend_penalty
        score += np.random.normal(0, 0.6)
        score = round(float(np.clip(score, 1.0, 10.0)), 1)

        sessions.append({
            'date': date.strftime('%Y-%m-%d'),
            'time_of_day': hour,
            'subject': subject,
            'duration_minutes': duration,
            'productivity_score': score,
        })

    return sessions


def seed():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    sessions = generate_sessions(80)
    with open(DATA_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sessions)
    print(f"✓ Seeded {len(sessions)} demo sessions → {DATA_PATH}")


if __name__ == '__main__':
    seed()
