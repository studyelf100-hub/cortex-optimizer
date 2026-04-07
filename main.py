"""
main.py — Pipeline entry point for Study Optimizer
"""

import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))

from logger import interactive_log
from preprocess import prepare, get_feature_columns
from model import train, load_model
from analysis import run_all
from scheduler import generate_schedule, print_schedule


def cmd_log():
    """Log a new study session."""
    interactive_log()


def cmd_train():
    """Preprocess data and train the model."""
    print("\nPreprocessing data...")
    df = prepare(save=True)
    print(f"\nTraining on {len(df)} sessions...\n")
    model, feature_cols, metrics = train(df, verbose=True)
    print(f"\n✓ Model trained. CV MAE: {metrics['mae_cv']:.3f}")


def cmd_analyze():
    """Generate all visualizations."""
    df = prepare(save=False)
    run_all(df)


def cmd_schedule():
    """Print optimal schedule recommendation."""
    model, meta = load_model()
    from preprocess import prepare
    df = prepare(save=False)
    subjects = df['subject'].unique().tolist()
    feature_cols = meta['feature_cols']
    result = generate_schedule(model, feature_cols, subjects)
    print_schedule(result)


def cmd_all():
    """Run full pipeline: preprocess → train → analyze → schedule."""
    cmd_train()
    cmd_analyze()
    cmd_schedule()


COMMANDS = {
    'log': (cmd_log, 'Log a new study session'),
    'train': (cmd_train, 'Preprocess and train the model'),
    'analyze': (cmd_analyze, 'Generate visualizations'),
    'schedule': (cmd_schedule, 'Show optimal schedule'),
    'all': (cmd_all, 'Run full pipeline'),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("\nStudy Optimizer — Usage:")
        for cmd, (_, desc) in COMMANDS.items():
            print(f"  python main.py {cmd:12s} {desc}")
        print()
        return

    cmd = sys.argv[1]
    COMMANDS[cmd][0]()


if __name__ == '__main__':
    main()
