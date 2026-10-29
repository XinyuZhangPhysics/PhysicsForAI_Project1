from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np
import torch
from torch.nn import functional as F


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def norm(t):
    return float(torch.linalg.vector_norm(t.double()))

def dot(a,b):
    return float((a.double()*b.double()).sum())

class Op:
    def __init__(self,h,p,g,coef=.25): self.h=h; self.p=p; self.g=g; self.coef=coef
    def ct(self,u): return -self.coef*(self.g.T@((self.h@u).unsqueeze(-1)*self.p))
    def c(self,m): return -self.coef*(self.h.T@(self.g*F.linear(self.p,m)).sum(-1))

def step(X,U,op,mu,lam,eta_x,eta_u):
    return ((1-eta_x*mu)*X+eta_x*op.ct(U),(1-eta_u*lam)*U+eta_u*op.c(X))

def close(a,b,rtol=2e-6,atol=1e-11):
    return math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    d=args.result_dir
    man=json.loads((d/'generic_local_dynamics_manifest.json').read_text())
    if man.get('status')!='COMPLETE': raise SystemExit('Cannot verify non-COMPLETE result')
    z=np.load(d/'generic_local_dynamics_operator.npz',allow_pickle=False)
    h=torch.from_numpy(z['h'].copy()); p=torch.from_numpy(z['phi'].copy()); g=torch.from_numpy(z['g'].copy()); op=Op(h,p,g)
    v=torch.from_numpy(z['v'].copy()); u=torch.from_numpy(z['u'].copy())
    Xdirs=z['X_directions']; Udirs=z['U_directions']; seeds=z['direction_seeds'].astype(int).tolist()
    mu=float(z['mu']); eta_x=float(z['eta_X']); eta_u=float(z['eta_U']); amps={'large':float(z['A_large']),'small':float(z['A_small'])}
    rows=list(csv.DictReader((d/'generic_local_dynamics_long.csv').open()))
    by={}
    for r in rows: by.setdefault((r['side'],int(r['direction']),r['amplitude']),[]).append(r)
    expected_arms=2*8*2
    if len(by)!=expected_arms: raise SystemExit(f'Expected {expected_arms} arms, found {len(by)}')
    checks=0; max_projection_error=0.0; max_norm_error=0.0
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    for (side,di,amp),rr in sorted(by.items()):
        rr=sorted(rr,key=lambda x:int(x['step']))
        if [int(x['step']) for x in rr] != list(range(int(man['horizon'])+1)): raise SystemExit(f'Noncontiguous steps {side}/{di}/{amp}')
        eps=-0.08 if side=='stable' else 0.08; lam=mu*(1-eps)
        A=amps[amp]; X=torch.from_numpy((A*Xdirs[di-1]).astype(np.float32)); U=torch.from_numpy((A*Udirs[di-1]).astype(np.float32))
        for n,r in enumerate(rr):
            if int(r['seed']) != seeds[di-1]: raise SystemExit(f'seed mismatch {side}/{di}/{amp}/{n}')
            if tsha(X)!=r['linear_X_sha256'] or tsha(U)!=r['linear_U_sha256']:
                raise SystemExit(f'linear recurrence hash mismatch {side}/{di}/{amp}/step{n}')
            raw=math.hypot(norm(X),norm(U)); recorded=float(r['linear_state_raw'])
            max_norm_error=max(max_norm_error,abs(raw-recorded))
            if not close(raw,recorded): raise SystemExit(f'linear norm mismatch {side}/{di}/{amp}/step{n}')
            for k in range(4):
                mk=dot(X,v[:,:,k]); uk=dot(U,u[:,k])
                max_projection_error=max(max_projection_error,abs(mk-float(r[f'm{k+1}_lin'])),abs(uk-float(r[f'u{k+1}_lin'])))
                if not close(mk,r[f'm{k+1}_lin']) or not close(uk,r[f'u{k+1}_lin']):
                    raise SystemExit(f'linear projection mismatch {side}/{di}/{amp}/step{n}/mode{k+1}')
            checks+=1
            if n<int(man['horizon']): X,U=step(X,U,op,mu,lam,eta_x,eta_u)
    out={'status':'PASS','verification':'independent recurrence reconstruction from saved matrix-free factorization and frozen directions','arms_verified':len(by),'rows_verified':checks,'max_abs_linear_projection_replay_error':max_projection_error,'max_abs_linear_norm_replay_error':max_norm_error,'no_outcome_filtering_checked_by_arm_count':True}
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__': main()
