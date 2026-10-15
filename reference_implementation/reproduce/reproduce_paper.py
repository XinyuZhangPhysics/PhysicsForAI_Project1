"""Reconstruct declared paper outputs from frozen numerical sufficient inputs."""
from pathlib import Path
import os,sys,csv,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.mplconfig'))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from moe_private_response import formation,native_tomography as native,q3
from verify_release import verify
DATA=ROOT/'data/derived';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def rows(n):return list(csv.DictReader((DATA/n).open()))
def read(n):return json.loads((DATA/n).read_text())
def close(a,b):
    if not np.allclose(a,b,rtol=2e-10,atol=2e-13):raise ValueError((a,b))
def table(n,rs):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
checks={'release':verify(ROOT)}
for receipt in read('q3_receipt_hashes.json'):
    p=DATA/'q3_receipts'/(receipt['arm']+'.json')
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    if h!=receipt['receipt_sha256']:raise ValueError('original receipt mismatch')
    r=json.loads(p.read_text())
    if r['arm']!=receipt['arm'] or r['status']!='COMPLETE' or r['nll_table_sha256']!=receipt['original_nll_sha256'] or r['base_parameter_updates']!=0:raise ValueError('completion receipt semantics')
checks['Q3_original_receipts_verified']=41
with np.load(DATA/'q3_losses.npz') as z:result=q3.endpoint(z['losses'])
expected=read('q3_expected.json')
for key in result:close(result[key],expected[key])
checks['Q3']=result
rs=[];splits=[];eigenvalues_verified=0
with np.load(DATA/'T1_41B_TOMOGRAPHY.npz') as b,np.load(DATA/'T2_5033_TOMOGRAPHY.npz') as t:
    with np.load(DATA/'native_control_basis.npz') as basis:
        for layer in [7,15]:
            controls=basis[f'L{layer}_controls_exact_common_projected'];close(controls@controls.T,b[f'L{layer}_H'])
            for archive in [b,t]:
                for probe in ['local','objective']:
                    prefix=f'L{layer}_{probe}_J'
                    close(native.physical_operator(archive[prefix+'_mean_B'],archive[f'L{layer}_H']),archive[prefix+'_physical'])
    for item,expect in zip(read('native_scalars.json'),read('native_expected.json')):
        prefix=f"L{item['layer']}_{item['probe']}_J";c=item['Q5033_mode1']/item['Q41_mode1']
        r=native.scalar_gate(b[prefix+'_physical'],t[prefix+'_physical'],c)
        close(r['scale_error'],expect['epsilon_scale']);close(r['shape_cosine'],expect['shape_cosine'])
        r.update(layer=item['layer'],probe=item['probe'],scale_estimator=c);rs.append(r)
for cp in [41,146,503,1740,5033]:
    name='T1_41B_TOMOGRAPHY.npz' if cp==41 else 'T2_5033_TOMOGRAPHY.npz' if cp==5033 else 'T3_INTERMEDIATE_TOMOGRAPHY.npz'
    with np.load(DATA/name) as z:
        for layer in [7,15]:
            for probe in ['local','objective']:
                prefix=('' if cp in [41,5033] else f'T{cp}_')+f'L{layer}_{probe}_J'
                eig=np.linalg.eigvalsh(z[prefix+'_physical'])[::-1]
                frozen=sorted([r for r in rows('TOMO_ENDPOINT_EIGENVALUES.csv') if int(r['checkpoint_B'])==cp and int(r['layer'])==layer and r['probe']==probe],key=lambda r:int(r['index']))
                if frozen:
                    close(eig,[float(r['eigenvalue']) for r in frozen]);eigenvalues_verified+=len(frozen)
                gate=native.split_gate(z[prefix+'_half0_physical'],z[prefix+'_half1_physical'])
                if not gate['stable']:raise ValueError('split-half gate failed')
                gate.update(checkpoint=cp,layer=layer,probe=probe);splits.append(gate)
table('native_gates.csv',rs);table('native_split_gates.csv',splits)
checks['native']={'scalar_gates':rs,'stable_operators':len(splits),'endpoint_eigenvalues_verified':eigenvalues_verified}
pred=rows('formation/P2J_FROZEN_CELL_PREDICTIONS.csv');mu=float(pred[0]['mu']);eta=float(pred[0]['eta_M'])
with np.load(DATA/'formation/spectrum.npz') as z:kappa=float(z['s'][0]);close(kappa,mu);ld=kappa*kappa/mu
rates=[]
for item in rows('formation/P2J_DISCRETE_KINETICS.csv'):
    eps=float(item['epsilon']);ratio=float(item['r_eta']);gp=float(formation.growth_rate(mu,kappa,ld*(1-eps),eta,eta*ratio));close(gp,float(item['gamma_predicted']))
    go=float(item['gamma_observed']);valid=item['rate_valid']=='True'
    rates.append(dict(experiment='primary',epsilon=eps,r_eta=ratio,zeta=formation.zeta(eta,eta*ratio,ld,mu),gamma_predicted=gp,gamma_fitted=go,valid=valid,relative_error=abs(go-gp)/abs(gp),sigma_norm_predicted=math.expm1(gp)/(eta*mu*eps),sigma_norm_fitted=math.expm1(go)/(eta*mu*eps)))
valid=[r for r in rates if r['valid']];observed=np.array([r['gamma_fitted'] for r in valid]);predicted=np.array([r['gamma_predicted'] for r in valid])
E=float(formation.relative_l2(observed,predicted));close(E,read('formation/expected.json')['kinetic_relative_l2_error'])
confirmation=read('formation/confirmation.json')
for r in confirmation['cells']:
    ratio=float(r['r_eta']);gp=float(formation.growth_rate(mu,kappa,ld*.76,eta,eta*ratio));close(gp,r['predicted_gamma']);go=float(r['gamma_mean'])
    rates.append(dict(experiment='confirmation',epsilon=.24,r_eta=ratio,zeta=ratio,gamma_predicted=gp,gamma_fitted=go,valid=r['RATE_VALID'],relative_error=abs(go-gp)/abs(gp),sigma_norm_predicted=math.expm1(gp)/(eta*mu*.24),sigma_norm_fitted=math.expm1(go)/(eta*mu*.24)))
co=[r for r in rates if r['experiment']=='confirmation'];Ec=float(formation.relative_l2(np.array([r['gamma_fitted'] for r in co]),np.array([r['gamma_predicted'] for r in co])));close(Ec,confirmation['direct_rate_relative_L2'])
table('formation_rates.csv',rates);table('zeta.csv',[r for r in rates if r['experiment']=='primary' and r['epsilon']==.08])
checks['formation']={'kappa':kappa,'mu':mu,'lambda_d':ld,'primary_valid_cells':len(valid),'primary_relative_L2':E,'confirmation_relative_L2':Ec}
mc=rows('MODE_CONVERSION_SIGNED_SUMMARY.csv')
for r in mc:close(float(r['measured_signed_coefficient'])-float(r['predicted_analytic_coefficient']),float(r['residual']))
six=[r for r in mc if r['sector']=='M' and float(r['s'])==1 and int(r['coordinate'])==0];table('paired_residuals.csv',six)
selection=rows('R63_MAIN_FIGURE_DATA.csv')
for r in selection:close(float(r['dot_B_actual'])-float(r['dot_B_prediction_A']),float(r['residual']))
checks['selection']={'verified_coordinate_records':len(selection),'max_absolute_closure_residual':max(abs(float(r['residual'])) for r in selection),'signed_mode_conversion_records':len(mc),'six_paired_residual_cells':len(six),'intervals':'frozen whole-seed paired-bootstrap intervals; not recomputed from marginal intervals'}
# Full-state diagnostics from per-case sufficient statistics, with archived stream bootstrap draws.
cases=rows('FINITE_TIME_THREE_WAY_CASES.csv');summaries=rows('FINITE_TIME_THREE_WAY_SUMMARY.csv')
with np.load(DATA/'FINITE_TIME_STREAM_BOOTSTRAP_DRAWS.npz') as z:bootstrap={k:z[k].copy() for k in ['primary','replication']}
finite=[]
for pop in ['primary','replication']:
    for sector in ['full_state','expert_private','router_private']:
        rr=sorted([r for r in cases if r['population']==pop and r['sector']==sector],key=lambda r:(int(r['stream']),int(r['arm'])))
        for r in rr:
            for label in ['sequence','seqmean','refmean']:close(float(r[label+'_absolute_error'])/float(r['target_increment_norm']),float(r[label+'_original_error']))
            close(float(r['refmean_original_error'])/float(r['sequence_original_error']),float(r['refmean_over_sequence']))
            close(float(r['seqmean_original_error'])/float(r['sequence_original_error']),float(r['seqmean_over_sequence']))
        for stat in ['sequence_original_error','seqmean_original_error','refmean_original_error','refmean_over_sequence','seqmean_over_sequence','refmean_over_seqmean']:
            v=np.array([float(r[stat]) for r in rr]);median=float(np.median(v));mat=v.reshape(6 if pop=='primary' else 2,15)
            draws=np.median(mat[bootstrap[pop]].reshape(len(bootstrap[pop]),-1),axis=1);lo,hi=np.quantile(draws,[.025,.975])
            es=[r for r in summaries if r['population']==pop and r['sector']==sector and r['statistic']==stat]
            if es:close(median,float(es[0]['paired_case_median']));close([lo,hi],[float(es[0]['ci_low']),float(es[0]['ci_high'])])
            finite.append(dict(population=pop,sector=sector,statistic=stat,paired_case_median=median,ci_low=float(lo),ci_high=float(hi)))
table('finite_time_diagnostics.csv',finite)
cum=rows('FINITE_TIME_CUMULANTS.csv');residual_errors=[]
for r in cum:
    rn=float(r['ordered_vs_seqmean_residual_norm']);cn=float(r['second_order_action_correction_norm']);cos=float(r['second_order_residual_alignment'])
    explained=2*cos*cn/rn-(cn/rn)**2;archived=float(r['second_order_fraction_residual_explained']);close(explained,archived);residual_errors.append(abs(explained-archived))
headline=next(r for r in finite if r['population']=='primary' and r['sector']=='full_state' and r['statistic']=='refmean_over_sequence')
checks['finite_time']={'case_sector_records':len(cases),'reproduced_aggregate_rows':len(finite),'reference_mean_over_ordered':headline['paired_case_median'],'cumulant_residual_records':len(cum),'max_residual_reconstruction_error':max(residual_errors),'scope':'per-case full-state sufficient-statistic reconstruction; projected cumulants supplied; raw full-state maps not redistributed'}
(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
from pathlib import Path as _P
import subprocess
subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_tables.py')],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_figures.py')],check=True)
print(json.dumps({'status':'PASS','formation':checks['formation'],'Q3':result,'finite_time':checks['finite_time']},indent=2))
