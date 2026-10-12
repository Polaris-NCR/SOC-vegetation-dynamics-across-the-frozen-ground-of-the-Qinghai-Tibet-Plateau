"""
Fig. 4 — Scatter plots of observed vs predicted SOC storage (RF+Hyperion
models), three target depths, pooled predictions from the 100 repeated
70:30 partitions.

Data:
  results2026/rf-hpy-scatter-points-0-{depth}cm.csv
      columns: y_test (observed), y_pred (predicted)

Design:
  Three panels (30/50/100 cm); points coloured by density (hexbin-style);
  1:1 dashed line + fitted regression line; R2 / RMSE / MAE annotated in
  each panel. Arial, size 10. Panel labels (a)-(c) upper-left.

Output: visualization_results/soc_prediction_scatter.png

Run: python code/plot_prediction_scatter.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

BASE = 'results2026'
OUT_DIR = 'visualization_results'
os.makedirs(OUT_DIR, exist_ok=True)

DEPTHS = [30, 50, 100]
DEPTH_LABELS = {30: '0-30 cm', 50: '0-50 cm', 100: '0-100 cm'}
PANEL = ['(a)', '(b)', '(c)']
YMAX_FILTER = 40.0


def load_points(depth):
    df = pd.read_csv(os.path.join(BASE, f'rf-hpy-scatter-points-0-{depth}cm.csv'))
    y = df['y_test'].values.astype(float)
    yhat = df['y_pred'].values.astype(float)
    m = np.isfinite(y) & np.isfinite(yhat) & (y <= YMAX_FILTER) & (y > 0)
    return y[m], yhat[m]


def metrics(y, yhat):
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae = np.mean(np.abs(y - yhat))
    return r2, rmse, mae


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5))
    fig.subplots_adjust(right=0.88, wspace=0.25)

    for i, depth in enumerate(DEPTHS):
        ax = axes[i]
        y, yhat = load_points(depth)
        r2, rmse, mae = metrics(y, yhat)

        # density scatter via hexbin
        hb = ax.hexbin(y, yhat, gridsize=40, cmap='viridis',
                       mincnt=1)
        ax.plot([0, YMAX_FILTER], [0, YMAX_FILTER], 'k--', lw=1, alpha=0.7)

        # fitted regression line
        k, b = np.polyfit(y, yhat, 1)
        xs = np.linspace(y.min(), y.max(), 50)
        ax.plot(xs, k * xs + b, 'r-', lw=1.5)

        # depth label in the x-axis title position
        ax.set_xlabel(f'Observed SOC at {DEPTH_LABELS[depth]} '
                      f'(kg C m$^{{-2}}$)')
        if i == 0:
            ax.set_ylabel('Predicted SOC (kg C m$^{{-2}}$)')
        # ax.text(0.97, 0.05,
        #         f'R$^2$ = {r2:.2f}\n'
        #         f'RMSE = {rmse:.2f} kg C m$^{{-2}}$\n'
        #         f'MAE = {mae:.2f} kg C m$^{{-2}}$',
        #         transform=ax.transAxes, fontsize=9,
        #         ha='right', va='bottom',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.text(0.97, 0.02,
            f'R$^2$ = {r2:.2f}\n'
            f'RMSE = {rmse:.2f} kg C m$^{{-2}}$\n'
            f'MAE = {mae:.2f} kg C m$^{{-2}}$',
            transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom')
        # panel label inside the frame, upper-left corner
        # ax.text(0.03, 0.97, PANEL[i], transform=ax.transAxes,
        #         fontsize=10, fontweight='bold', va='top', ha='left')
        ax.text(0.03, 0.97, PANEL[i], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')
        ax.set_xlim(0, YMAX_FILTER)
        ax.set_ylim(0, YMAX_FILTER)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    # colourbar in its own axes to avoid overlap with panels
    cbar_ax = fig.add_axes([0.905, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(hb, cax=cbar_ax,
                        label='Point density (counts per bin)')
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.outline.set_visible(False)
    cbar.ax.set_frame_on(False)
    cbar_ax.patch.set_alpha(0)

    out = os.path.join(OUT_DIR, 'soc_prediction_scatter.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
