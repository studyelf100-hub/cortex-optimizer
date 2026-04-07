"""
logger.py — Study session input and persistence
"""

import csv
import os
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sessions.csv')
FIELDNAMES = ['date', 'time_of_day', 'subject', 'duration_minutes', 'productivity_score']

SUBJECTS = ['Mathematics', 'Physics', 'Programming', 'Chemistry', 'Biology',
            'History', 'Literature', 'Languages', 'Economics', 'Other']


def ensure_file():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_session(time_of_day: int, subject: str, duration_minutes: int,
                productivity_score: float, date: str = None):
    """
    Log a single study session.

    Args:
        time_of_day: Hour of day (0–23)
        subject: Subject studied
        duration_minutes: Duration in minutes
        productivity_score: Self-rated score (1–10)
        date: ISO date string (defaults to today)
    """
    ensure_file()
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    row = {
        'date': date,
        'time_of_day': time_of_day,
        'subject': subject,
        'duration_minutes': duration_minutes,
        'productivity_score': productivity_score,
    }

    with open(DATA_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

    print(f"✓ Session logged: {subject} at {time_of_day}:00 for {duration_minutes}min → score {productivity_score}")
    return row


def interactive_log():
    """CLI interface for logging a session."""
    print("\n═══════════════════════════════")
    print("    STUDY OPTIMIZER — LOG SESSION")
    print("═══════════════════════════════\n")

    print("Subjects:")
    for i, s in enumerate(SUBJECTS, 1):
        print(f"  {i}. {s}")

    while True:
        try:
            subject_idx = int(input("\nSubject number: ")) - 1
            if 0 <= subject_idx < len(SUBJECTS):
                subject = SUBJECTS[subject_idx]
                break
        except ValueError:
            pass
        print("Invalid choice.")

    while True:
        try:
            time_of_day = int(input("Start hour (0–23): "))
            if 0 <= time_of_day <= 23:
                break
        except ValueError:
            pass
        print("Enter a number between 0 and 23.")

    while True:
        try:
            duration = int(input("Duration (minutes): "))
            if duration > 0:
                break
        except ValueError:
            pass
        print("Enter a positive number.")

    while True:
        try:
            score = float(input("Productivity score (1–10): "))
            if 1.0 <= score <= 10.0:
                break
        except ValueError:
            pass
        print("Enter a number between 1 and 10.")

    return log_session(time_of_day, subject, duration, score)


if __name__ == '__main__':
    interactive_log()
