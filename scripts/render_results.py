"""Regenerate a static figure from public aggregate results only."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT/'results/development_to_holdout_comparison.csv').open(encoding='utf-8-sig') as f:
        rows = {r['metric']: r for r in csv.DictReader(f)}
    metrics = [('Stable reduction', 'Stable volatility reduction'),
               ('High response retention', 'High-change response retained')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['#64748b', '#087f8c']
    for ax, (key, title) in zip(axes, metrics):
        values = [float(rows[key][s])*100 for s in ['development', 'holdout']]
        bars = ax.bar(['Development OOF', 'Locked holdout'], values, color=colors, width=.55)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=5, fontsize=13)
        ax.set_ylim(0, 105 if 'High' in key else 40)
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_ylabel('Percent (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('SG-TR: development effects reproduced in a same-source customer holdout', fontsize=14)
    fig.text(.5, .02, '15,000 development customers | 30,000 holdout customers | raw-score temporal metrics', ha='center', fontsize=10)
    fig.tight_layout(rect=[0,.07,1,.93])
    (ROOT/'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT/'assets/effect_comparison.png', dpi=160, facecolor='white')
    plt.close(fig)
    print('Rendered assets/effect_comparison.png from aggregate tables.')


if __name__ == '__main__':
    main()
