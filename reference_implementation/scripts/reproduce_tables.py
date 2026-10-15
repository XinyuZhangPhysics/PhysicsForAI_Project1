"""Export reconstructed CSV tables as standalone LaTeX tabulars."""
from pathlib import Path
import csv
root=Path(__file__).resolve().parents[1]/'results'
def escape(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
columns={'formation_rates':['experiment','epsilon','r_eta','gamma_predicted','gamma_fitted','valid'],
         'native_gates':['layer','probe','scale_estimator','scale_error','shape_cosine','adequate'],
         'native_split_gates':['checkpoint','layer','probe','shape_cosine','relative_deff_difference','stable'],
         'paired_residuals':['family','g','predicted_analytic_coefficient','measured_signed_coefficient','residual','residual_ci_low','residual_ci_high'],
         'finite_time_diagnostics':['population','sector','statistic','paired_case_median','ci_low','ci_high']}
def fmt(s):
    try:return f'{float(s):.8g}'
    except ValueError:return s
for name,keys in columns.items():
    data=list(csv.DictReader((root/(name+'.csv')).open()))
    rows=[keys]+[[fmt(r[k]) for k in keys] for r in data]
    lines=[r'\begin{tabular}{'+'l'*len(keys)+'}',r'\hline']
    for row in rows:lines.append(' & '.join(escape(x) for x in row)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}']);(root/(name+'.tex')).write_text('\n'.join(lines)+'\n')
