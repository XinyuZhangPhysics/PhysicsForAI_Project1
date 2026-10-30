"""Frozen adjudication and report packaging; no scientific parameter fitting."""
import csv,json,hashlib,zipfile,time
from pathlib import Path
import numpy as np
from preflight import R,W,put,sha

def adjudicate():
 data=json.loads((R/'T3_BANK_B_PRIMARY_SUMMARY.json').read_text());p=data['primary'];low=data['all_amplitudes'][0];high=data['all_amplitudes'][2]
 def chk(v,rule,passed):return {'observed':v,'threshold':rule,'pass':bool(passed)}
 # The protocol's qualitative catastrophe check: a sign pair should retain the direction of an odd response.
 pairmin=min(x['sign_pair_anticorrelation_cosine'] for x in p['sign_pair_diagnostics'])
 checks={'E_full':chk(p['E_full'],'<=0.50',p['E_full']<=.50),'R':chk(p['R'],'<=0.75',p['R']<=.75),'median_C_full':chk(p['median_C_full'],'>=0.90',p['median_C_full']>=.90),'direction_pairs_full_better':chk(p['direction_pairs_full_better'],'>=10/12',p['direction_pairs_full_better']>=10),'bootstrap_R_upper':chk(p['R_bootstrap_ci95'][1],'<1.0',p['R_bootstrap_ci95'][1]<1),'R_a001':chk(low['R'],'<1.0',low['R']<1),'median_C_full_a001':chk(low['median_C_full'],'>0',low['median_C_full']>0),'sign_pair_locality':chk(pairmin,'no catastrophic sign reversal: response(+a) and -response(-a) retain positive alignment',pairmin>0)}
 plus={'median_support_turnover':chk(p['median_support_turnover'],'<=0.01',p['median_support_turnover']<=.01),'p95_support_turnover':chk(p['p95_support_turnover'],'<=0.05',p['p95_support_turnover']<=.05),'R_a001':chk(low['R'],'<1.0',low['R']<1),'R_a004':chk(high['R'],'<1.0',high['R']<1)}
 A=all(c['pass'] for c in checks.values());Ap=A and all(c['pass'] for c in plus.values())
 if Ap:status='GRADE_A_PLUS_SOFTEN_TOPK_QUALIFICATION'
 elif A:status='GRADE_A_DELETE_NATIVE_QUALIFICATION'
 elif p['R']<1:status='GRADE_B_POSITIVE_BUT_INSUFFICIENT'
 else:status='GRADE_C_NEGATIVE'
 metrics={k:v for k,v in p.items() if k not in ['pairs','sign_pair_diagnostics']};metrics.update(R_a001=low['R'],R_a004=high['R'])
 put('NMIX_ADJUDICATION.json',{'protocol_id':'Project1_NMIX_v1','run_revision':4,'final_status':status,'primary_amplitude':.02,'metrics':metrics,'grade_A_checks':checks,'grade_A_plus_checks':plus,'sign_pair_diagnostics':p['sign_pair_diagnostics'],'scientific_evaluation_occurred':True,'notes':['The prior missing-input blocked run is not a negative scientific result.','The qualitative no-catastrophe test is evaluated as retained sign-pair odd-response alignment; raw asymmetry values are reported.']})
 return status,metrics,checks,plus

def package():
 # Recheck every pre-primary source/config/input identity and primary output seal before packaging.
 for sealname in ['T0_PROTOCOL_SEAL.json','audit/PRIMARY_RESULTS_SEAL.json']:
  if not (R/sealname).is_file():continue
  seal=json.loads((R/sealname).read_text())
  for name,h in seal['files'].items():assert sha(R/name)==h,(sealname,name)
 files=[p for p in sorted(R.rglob('*')) if p.is_file() and p.name!='RESULT_SHA256SUMS.txt' and '__pycache__' not in p.parts]
 (R/'RESULT_SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in files)+'\n')
 z=R.parent/'Project1_NMIX_v1_RESULTS.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for p in files+[R/'RESULT_SHA256SUMS.txt']:archive.write(p,p.relative_to(R.parent))
 with zipfile.ZipFile(z) as archive:assert archive.testzip() is None
 h=sha(z);z.with_suffix('.zip.sha256').write_text(f'{h}  {z.name}\n');print(json.dumps({'ZIP':str(z),'SHA256':h},indent=2))

if __name__=='__main__':
 status,m,c,plus=adjudicate()
 print(json.dumps({'final_status':status,'metrics':m},indent=2))
 # Reporting text is supplied from the certified raw summaries; no new experiment follows adjudication.