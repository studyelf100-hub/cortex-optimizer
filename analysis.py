"""
analysis.py — Pattern detection and visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

FIGURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def set_style():
    """Apply custom dark aesthetic."""
    plt.rcParams.update({
        'figure.facecolor': '#0d0d0f',
        'axes.facecolor': '#13131a',
        'axes.edgecolor': '#2a2a35',
        'axes.labelcolor': '#c8c8d8',
        'axes.titlecolor': '#e8e8f0',
        'xtick.color': '#888899',
        'ytick.color': '#888899',
        'text.color': '#c8c8d8',
        'grid.color': '#2a2a35',
        'grid.linewidth': 0.6,
        'font.family': 'monospace',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


ACCENT = '#7c6af7'
ACCENT2 = '#f76a8c'
ACCENT3 = '#6af7c4'
PALETTE = [ACCENT, ACCENT2, ACCENT3, '#f7c46a', '#6ac4f7']


def plot_productivity_by_hour(df: pd.DataFrame, save: bool = True):
    """Productivity score vs hour of day."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d0d0f')

    hourly = df.groupby('time_of_day')['productivity_score'].agg(['mean', 'std', 'count'])
    hourly = hourly.reset_index()

    ax.fill_between(hourly['time_of_day'],
                    hourly['mean'] - hourly['std'].fillna(0),
                    hourly['mean'] + hourly['std'].fillna(0),
                    alpha=0.15, color=ACCENT)
    ax.plot(hourly['time_of_day'], hourly['mean'], color=ACCENT,
            linewidth=2.5, marker='o', markersize=5)

    peak_row = hourly.loc[hourly['mean'].idxmax()]
    ax.axvline(peak_row['time_of_day'], color=ACCENT, alpha=0.3, linestyle='--')
    ax.annotate(f"Peak: {int(peak_row['time_of_day'])}:00\n({peak_row['mean']:.1f})",
                xy=(peak_row['time_of_day'], peak_row['mean']),
                xytext=(peak_row['time_of_day'] + 1, peak_row['mean'] - 0.5),
                color=ACCENT, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2))

    ax.set_xlabel('Hour of Day', labelpad=8)
    ax.set_ylabel('Mean Productivity (1–10)', labelpad=8)
    ax.set_title('PRODUCTIVITY BY HOUR', pad=15, fontsize=13, fontweight='bold',
                 color='#e8e8f0')
    ax.set_xlim(0, 23)
    ax.set_ylim(1, 10.5)
    ax.grid(True, axis='y')
    plt.tight_layout()

    path = os.path.join(FIGURES_PATH, 'productivity_by_hour.png')
    if save:
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✓ Saved: {path}")
    return fig


def plot_subject_breakdown(df: pd.DataFrame, save: bool = True):
    """Horizontal bar chart of mean productivity per subject."""
    set_style()
    subj = df.groupby('subject')['productivity_score'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, len(subj) * 0.7)))
    fig.patch.set_facecolor('#0d0d0f')

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(subj))]
    bars = ax.barh(subj.index, subj.values, color=colors, height=0.6, alpha=0.85)

    for bar, val in zip(bars, subj.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9, color='#c8c8d8')

    ax.set_xlabel('Mean Productivity Score', labelpad=8)
    ax.set_title('PERFORMANCE BY SUBJECT', pad=15, fontsize=13, fontweight='bold', color='#e8e8f0')
    ax.set_xlim(0, 11)
    ax.grid(True, axis='x')
    plt.tight_layout()

    path = os.path.join(FIGURES_PATH, 'subject_breakdown.png')
    if save:
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✓ Saved: {path}")
    return fig


def plot_fatigue_curve(df: pd.DataFrame, save: bool = True):
    """Productivity vs session duration (fatigue curve)."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d0d0f')

    df['duration_bucket'] = pd.cut(df['duration_minutes'],
                                   bins=[0, 30, 60, 90, 120, 150, 180, 360],
                                   labels=['≤30', '31–60', '61–90', '91–120',
                                           '121–150', '151–180', '>180'])
    fatigue = df.groupby('duration_bucket', observed=True)['productivity_score'].mean()

    ax.fill_between(range(len(fatigue)), fatigue.values, alpha=0.15, color=ACCENT2)
    ax.plot(range(len(fatigue)), fatigue.values, color=ACCENT2,
            linewidth=2.5, marker='s', markersize=6)
    ax.set_xticks(range(len(fatigue)))
    ax.set_xticklabels(fatigue.index, rotation=15)
    ax.set_xlabel('Session Duration (minutes)', labelpad=8)
    ax.set_ylabel('Mean Productivity', labelpad=8)
    ax.set_title('FATIGUE CURVE — DURATION VS PERFORMANCE', pad=15,
                 fontsize=13, fontweight='bold', color='#e8e8f0')
    ax.set_ylim(1, 10.5)
    ax.grid(True, axis='y')
    plt.tight_layout()

    path = os.path.join(FIGURES_PATH, 'fatigue_curve.png')
    if save:
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✓ Saved: {path}")
    return fig


def plot_heatmap(df: pd.DataFrame, save: bool = True):
    """Subject × hour heatmap of mean productivity."""
    set_style()
    pivot = df.pivot_table(values='productivity_score', index='subject',
                           columns='time_of_day', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.7)))
    fig.patch.set_facecolor('#0d0d0f')

    cmap = sns.color_palette("rocket", as_cmap=True)
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt='.1f',
                linewidths=0.5, linecolor='#1a1a22',
                cbar_kws={'label': 'Mean Productivity'})
    ax.set_title('PERFORMANCE HEATMAP — SUBJECT × HOUR', pad=15,
                 fontsize=13, fontweight='bold', color='#e8e8f0')
    ax.set_xlabel('Hour of Day', labelpad=8)
    ax.set_ylabel('')
    plt.tight_layout()

    path = os.path.join(FIGURES_PATH, 'heatmap.png')
    if save:
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✓ Saved: {path}")
    return fig


def run_all(df: pd.DataFrame):
    plot_productivity_by_hour(df)
    plot_subject_breakdown(df)
    plot_fatigue_curve(df)
    plot_heatmap(df)
    print("\n✓ All visualizations saved.")
