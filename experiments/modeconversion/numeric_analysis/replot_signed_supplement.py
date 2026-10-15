# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys
from pathlib import Path
ROOT = Path(_p1_path('path_fd01557d84f7'))
OUT = ROOT / 'outputs/v285M_archive_completion'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/mplconfig')
import pandas as pd
sys.path.append(_p1_path('path_d4fe3b336f94'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
summary = pd.read_csv(OUT / 'MODE_CONVERSION_SIGNED_SUMMARY.csv', float_precision='round_trip')
for sector in ['C', 'M']:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    handles = []
    for idx, (ax, (family, g)) in enumerate(zip(axes.flat, [(f, g) for f in ['S1', 'S2'] for g in [2, 4, 8]])):
        for coord in range(12):
            x = summary[(summary.family == family) & (summary.g == g) & (summary.sector == sector) & (summary.coordinate == coord)].sort_values('s')
            color = plt.cm.tab20(coord)
            line = ax.plot(x.s, x.predicted_analytic_coefficient, color=color, lw=0.9, label=f'Coordinate {coord}')[0]
            ax.scatter(x.s, x.measured_signed_coefficient, color=color, s=5)
            if idx == 0:
                handles.append(line)
        ax.axhline(0, color='.7', lw=0.6)
        ax.set_title(f'{family}, g={g}, sector {sector}')
        ax.set_xlabel('s')
        ax.set_ylabel('Signed coefficient')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=8, frameon=False)
    fig.savefig(OUT / f'FIG_MODE_CONVERSION_SIGNED_{sector}_ALL_COORDINATES.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'FIG_MODE_CONVERSION_SIGNED_{sector}_ALL_COORDINATES.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
