from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'evidence/generic_local_dynamics/generic_local_dynamics_long.csv'
MAN = ROOT / 'evidence/generic_local_dynamics/generic_local_dynamics_manifest.json'
OUT = Path(__file__).resolve().parent

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

df=pd.read_csv(SRC)
man=json.loads(MAN.read_text())
etaX=float(man['eta_X']); etaU=float(man['eta_U']); mu=float(man['mu'])
rows=[]
for j in range(1,5):
    X1=[]; Y1=[]; X2=[]; Y2=[]
    for _,g in df.groupby(['side','direction','amplitude'], sort=False):
        g=g.sort_values('step')
        lam=float(g['lambda_'].iloc[0])
        m=g[f'm{j}_lin'].to_numpy(float); u=g[f'u{j}_lin'].to_numpy(float)
        X1.append(u[:-1]); Y1.append(m[1:]-(1-etaX*mu)*m[:-1])
        X2.append(m[:-1]); Y2.append(u[1:]-(1-etaU*lam)*u[:-1])
    x1=np.concatenate(X1); y1=np.concatenate(Y1)
    x2=np.concatenate(X2); y2=np.concatenate(Y2)
    rho_x=float((x1@y1)/(etaX*(x1@x1)))
    rho_u=float((x2@y2)/(etaU*(x2@x2)))
    pred=np.concatenate([etaX*x1,etaU*x2]); yy=np.concatenate([y1,y2])
    rho=float((pred@yy)/(pred@pred))
    rel=float(np.linalg.norm(yy-rho*pred)/max(np.linalg.norm(yy),1e-300))
    pr=man['pair_residuals'][j-1]
    rows.append(dict(pair=j,rho_reconstructed=rho,rho_from_X_equation=rho_x,rho_from_U_equation=rho_u,
                     rho_over_rho1=np.nan,recurrence_relative_L2=rel,
                     archived_Cv_relative_residual=float(pr['c_v']),archived_CtU_relative_residual=float(pr['ct_u'])))
rho1=rows[0]['rho_reconstructed']
for r in rows: r['rho_over_rho1']=r['rho_reconstructed']/rho1
pd.DataFrame(rows).to_csv(OUT/'mixed_spectrum_reconstruction.csv', index=False)
report={
 'status':'POSTHOC_DETERMINISTIC_RECONSTRUCTION_FROM_SAVED_LINEAR_COEFFICIENTS',
 'source_file':str(SRC.relative_to(ROOT)), 'source_sha256':sha256(SRC),
 'manifest_file':str(MAN.relative_to(ROOT)), 'manifest_sha256':sha256(MAN),
 'eta_X':etaX,'eta_U':etaU,'mu':mu,
 'formula':'For each saved transition and singular pair j: m_{n+1}-(1-eta_X mu)m_n = eta_X rho_j u_n; u_{n+1}-(1-eta_U lambda)u_n = eta_U rho_j m_n. rho_j is a combined least-squares reconstruction over all 32 arms and 28 transitions.',
 'important_scope':'This reconstructs singular-value magnitudes already encoded in the saved frozen-map recurrence; it is not a replacement for the original spectrum NPZ, whose SHA-256 is retained in generic_local_dynamics_manifest.json.',
 'values':rows,
 'leading_gap':{'rho2_over_rho1':rows[1]['rho_over_rho1'],'rho1_over_rho2':1/rows[1]['rho_over_rho1']},
 'crosscheck_rho1_minus_manifest_mu':rows[0]['rho_reconstructed']-mu
}
(OUT/'mixed_spectrum_reconstruction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
