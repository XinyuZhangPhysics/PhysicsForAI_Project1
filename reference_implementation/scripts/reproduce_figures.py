"""Re-render numerical figures from reconstructed outputs and an external derived-data bundle."""
from pathlib import Path
import os,csv,json
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('PROJECT1_DATA_DIR', str(ROOT/'data/derived'))).expanduser().resolve()
if not DATA.exists():
    raise FileNotFoundError("Set PROJECT1_DATA_DIR to the external project-owned derived-data bundle.")
OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rows(p):return list(csv.DictReader(p.open()))
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig(FIG/(name+'.png'),dpi=160);plt.close(fig)
rr=rows(OUT/'formation_rates.csv');values=np.array([[100*next(float(r['relative_error']) for r in rr if float(r['epsilon'])==eps and float(r['r_eta'])==z and r['experiment']==('confirmation' if eps==.24 else 'primary')) for z in [.25,1,4]] for eps in [-.24,-.08,.08,.24]])
fig,ax=plt.subplots(figsize=(5.5,3.7));im=ax.imshow(values,cmap='Blues');ax.set(xticks=range(3),xticklabels=['0.25','1','4'],yticks=range(4),yticklabels=['-.24 (P)','-.08 (P)','+.08 (P)','+.24 (C)'],xlabel='Mobility ratio',ylabel='Offset and run')
for i in range(4):
    for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}%',ha='center',va='center')
fig.colorbar(im,ax=ax,label='Absolute relative log-rate error (%)');save(fig,'formation_closure')
rr=rows(OUT/'zeta.csv');fig,ax=plt.subplots(figsize=(5.5,3.7));x=np.logspace(-2,2,300);ax.plot(x,x/(1+x),label='Near-threshold limit');ax.scatter([float(r['zeta']) for r in rr],[float(r['sigma_norm_predicted']) for r in rr],label='Exact discrete');ax.scatter([float(r['zeta']) for r in rr],[float(r['sigma_norm_fitted']) for r in rr],marker='x',label='Frozen fit');ax.set(xscale='log',xlabel='zeta',ylabel='Normalized per-update increment');ax.legend();save(fig,'zeta_crossover')
rr=rows(DATA/'MODE_CONVERSION_SIGNED_SUMMARY.csv');fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,family in zip(axes,['S1','S2']):
    for g in [2,4,8]:
        rs=sorted([r for r in rr if r['family']==family and r['g']==str(g) and r['sector']=='M' and r['coordinate']=='0'],key=lambda r:float(r['s']))
        x=[float(r['s']) for r in rs];y=np.array([float(r['measured_signed_coefficient']) for r in rs]);lo=np.array([float(r['measured_signed_coefficient_ci_low']) for r in rs]);hi=np.array([float(r['measured_signed_coefficient_ci_high']) for r in rs])
        ax.plot(x,[float(r['predicted_analytic_coefficient']) for r in rs],label=f'g={g} prediction');ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='.',capsize=2)
    ax.set(title=family,xlabel='Signed background s');ax.legend(fontsize=8)
axes[0].set_ylabel('Signed coefficient, mixed sector, coordinate 0');save(fig,'mode_conversion_signed')
rr=rows(OUT/'native_gates.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.6));labels=[f"L{r['layer']}/{r['probe']}" for r in rr]
for ax,key,threshold in zip(axes,['scale_error','shape_cosine'],[.20,.95]):
    ax.bar(range(4),[float(r[key]) for r in rr]);ax.axhline(threshold,color='red',linestyle='--');ax.set(xticks=range(4),xticklabels=labels,ylabel=key.replace('_',' '));ax.tick_params(axis='x',rotation=35)
save(fig,'native_gates')
rr=rows(OUT/'finite_time_diagnostics.csv');fig,ax=plt.subplots(figsize=(6,3.6));keys=['refmean_original_error','seqmean_original_error','sequence_original_error'];rs=[next(r for r in rr if r['population']=='primary' and r['sector']=='full_state' and r['statistic']==k) for k in keys];y=np.array([float(r['paired_case_median']) for r in rs]);lo=np.array([float(r['ci_low']) for r in rs]);hi=np.array([float(r['ci_high']) for r in rs]);ax.bar(['Reference mean','Realized mean','Ordered'],y);ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='none',color='black',capsize=4);ax.set(yscale='log',ylabel='Full-state relative closure error');save(fig,'finite_time_closure')
