"""
Fig. S2 - 10x8 reconstruction visualization grid.

Layout:
  columns 1-6: first band of the Blue, Green, Red, NIR, SWIR1, SWIR2 branches
  rows: five selected patches, alternating ground truth / reconstruction
  columns 7-8: patch-mean spectrum per patch (ground truth dark blue circles,
               reconstruction light blue squares), one double-height panel
               per patch; spectral style follows pre_spectral_full57.png
               (code/pre_vis_UNet.py): wavelength axis in um, pastel branch
               shading with branch labels above the axes.

Input arrays (must already exist in the namespace, shapes (N, n_bands, 64, 64)):
  EO_blue_gt ... EO_swir2_gt      ground-truth Hyperion per branch
  pred_blue ... pred_swir2        KGDF reconstruction per branch
(Landsat_*_gt arrays are not used by this figure.)

Alignment rules enforced: the six branches must share the same (64, 64)
spatial grid; spectra are patch means over identical pixels for GT and pred.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

BRANCHES = [('blue', 7), ('green', 8), ('red', 6),
            ('nir', 5), ('swir1', 11), ('swir2', 19)]
BRANCH_LABEL = {'blue': 'Blue', 'green': 'Green', 'red': 'Red',
                'nir': 'NIR', 'swir1': 'SWIR1', 'swir2': 'SWIR2'}

PATCH_IDS = [31, 24, 30, 14, 26]           # 5 of the patches; state in caption
N_PATCHES = len(PATCH_IDS)

# Branch wavelength ranges (um) and channel counts (pre_vis_UNet.py).
# WAVELENGTHS is generated from these; override with exact centres if needed.
BAND_WAVELENGTHS = {
    'blue':  (0.457, 0.518, 7),
    'green': (0.528, 0.609, 8),
    'red':   (0.620, 0.681, 6),
    'nir':   (0.844, 0.879, 5),
    'swir1': (1.558, 1.659, 11),
    'swir2': (2.113, 2.294, 19),
}
WAVELENGTHS = None

SHADE_COLORS = ['#e8f0fe', '#e6f4ea', '#fce8e6', '#fff3e0', '#f3e5f5', '#e0f2f1']
GT_COLOR = '#1a3a5c'        # ground truth (dark blue)
REC_COLOR = '#4a90c4'       # reconstruction (light blue)

OUT = 'visualization_results/fig_s2_reconstruction_grid.png'


def collect_arrays():
    """Gather the per-branch GT/pred arrays from the current namespace."""
    gt, pred = {}, {}
    for name, _ in BRANCHES:
        g = globals().get(f'EO_{name}_gt')
        p = globals().get(f'pred_{name}')
        if g is None or p is None:
            raise NameError(
                f'EO_{name}_gt / pred_{name} not found in the namespace; '
                f'define all 12 arrays before running.')
        assert g.ndim == 4 and g.shape[0] > max(PATCH_IDS), \
            f'EO_{name}_gt: unexpected shape {g.shape}'
        assert p.shape == g.shape, \
            f'{name}: GT/pred shape mismatch {g.shape} vs {p.shape}'
        gt[name] = g[PATCH_IDS]
        pred[name] = p[PATCH_IDS]
        print(f'{name}: bands={g.shape[1]}, GT/pred shapes OK')
    return gt, pred


def branch_x_ranges():
    """(start, end) index of each branch in the concatenated 56-band axis."""
    ranges, pos = [], 0
    for name, nb in BRANCHES:
        ranges.append((name, pos, pos + nb))
        pos += nb
    return ranges


def make_xaxis():
    if WAVELENGTHS is not None:
        assert len(WAVELENGTHS) == 56
        x = np.array(WAVELENGTHS, dtype=float)
    else:
        wl = []
        for name, _ in BRANCHES:
            wl_s, wl_e, n_ch = BAND_WAVELENGTHS[name]
            wl.extend(np.linspace(wl_s, wl_e, n_ch))
        x = np.array(wl)
    return x, 'Wavelength (μm)'


gt, pred = collect_arrays()
vlims = {}
for name, _ in BRANCHES:
    v = gt[name][:, 0, :, :]           # first band, 5 patches
    vlims[name] = (np.percentile(v, 2), np.percentile(v, 98))

x, xlabel = make_xaxis()
ranges = branch_x_ranges()

fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(2 * N_PATCHES, 8,
                      width_ratios=[1, 1, 1, 1, 1, 1, 3, 1],
                      left=0.05, right=0.99, top=0.98, bottom=0.05,
                      wspace=0.18, hspace=0.18)

axes = [[fig.add_subplot(gs[r, c]) for c in range(6)]
        for r in range(2 * N_PATCHES)]

# ---- columns 1-6: first-band images, GT / recon alternating ----
for r in range(2 * N_PATCHES):
    k, is_gt = divmod(r, 2)
    src = gt if is_gt else pred
    for c, (name, _) in enumerate(BRANCHES):
        ax = axes[r][c]
        img = src[name][k, 0, :, :]
        ax.imshow(img, vmin=vlims[name][0], vmax=vlims[name][1])
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f'Scene {k + 1}\n{"Ground Truth" if is_gt else "Reconstructed"}',
                          rotation=0, ha='right', va='center', fontsize=9)
        if r == 2 * N_PATCHES - 1:
            ax.set_xlabel(f'{BRANCH_LABEL[name]}\n({x[ranges[c][1]]:.3f} μm)',
                          fontsize=9)

# ---- columns 7-8: patch-mean spectra, one double-height panel per patch ----
spec_axes = [fig.add_subplot(gs[2 * k:2 * k + 2, 6:8])
             for k in range(N_PATCHES)]

for k, ax in enumerate(spec_axes):
    # shade branch ranges and label branches above the axes
    for (name, s, e), col in zip(ranges, SHADE_COLORS):
        ax.axvspan(x[s], x[e - 1], alpha=0.25, color=col, zorder=0)
        ax.text((x[s] + x[e - 1]) / 2, 1.02, BRANCH_LABEL[name],
                ha='center', fontsize=9, fontweight='bold', color='gray',
                transform=ax.get_xaxis_transform())
    for is_gt, src, colour, marker in [
            (True, gt, GT_COLOR, 'o'), (False, pred, REC_COLOR, 's')]:
        spec = np.concatenate(
            [src[name][k].reshape(src[name][k].shape[0], -1).mean(axis=1)
             for name, _ in BRANCHES])
        ax.plot(x, spec, marker=marker, ms=3.5, color=colour, lw=1.4,
                label='Ground Truth' if is_gt else 'Reconstructed')
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticklabels([])          # keep only the y-axis reflectance values
    if k == 0:
        ax.legend(fontsize=10, frameon=False, loc='upper right')
    if k == N_PATCHES - 1:
        ax.set_xlabel(xlabel)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300)
plt.close(fig)
print(f'Saved: {OUT}')
