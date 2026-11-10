from pathlib import Path
import argparse,json
import numpy as np
N=50000;SEED=2026110901
def haar(rng,n=6):
 a=rng.standard_normal((n,n));q,r=np.linalg.qr(a);s=np.where(np.diag(r)>=0,1.,-1.);return q*s
def signed_perm(rng,n=6):
 p=rng.permutation(n);s=rng.choice(np.array([-1.,1.]),size=n);q=np.zeros((n,n));q[np.arange(n),p]=s;return q
def blockdiag(h):
 k=h.copy();k[:6,6:]=0;k[6:,:6]=0;return k
def error(h,z,y):return float(np.linalg.norm(y-z@h.T)/np.linalg.norm(y))
def family(h,z,y,name,rng):
 m=h[6:,:6];sv=np.linalg.svd(m,compute_uv=False);vals=np.empty(N)
 for i in range(N):
  if name=="both_haar":mn=haar(rng)@m@haar(rng).T
  elif name=="router_haar":mn=haar(rng)@m
  elif name=="expert_haar":mn=m@haar(rng).T
  elif name=="signed_permutation":mn=signed_perm(rng)@m@signed_perm(rng).T
  else:raise ValueError(name)
  k=h.copy();k[6:,:6]=mn;k[:6,6:]=mn.T
  assert np.max(np.abs(np.linalg.svd(mn,compute_uv=False)-sv))<=1e-12
  vals[i]=error(k,z,y)
 return vals
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--hessian",required=True);ap.add_argument("--responses",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
 h=np.load(args.hessian);d=np.load(args.responses);z=d["states"];y=d["responses"];mask=np.isclose(np.linalg.norm(z,axis=1),.02,rtol=0,atol=1e-12);z=z[mask];y=y[mask]
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(SEED);et=error(h,z,y);eb=error(blockdiag(h),z,y);summary={"seed":SEED,"draws":N,"E_true":et,"E_blockdiag":eb,"families":{}}
 for name in ("both_haar","router_haar","expert_haar","signed_permutation"):
  vals=family(h,z,y,name,rng);np.save(out/f"{name}_errors.npy",vals);summary["families"][name]={"rank":(1+int(np.count_nonzero(vals<=et)))/(N+1),"min":float(vals.min()),"q01":float(np.quantile(vals,.01)),"median":float(np.median(vals)),"max":float(vals.max())}
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
