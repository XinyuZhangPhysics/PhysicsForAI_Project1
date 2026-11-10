from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
COMMON=ROOT/"common"
SOURCE=ROOT/"source_to_heldout"/"reference"
def load(path):return json.loads(path.read_text())
def bd(H):
 K=H.copy();K[:6,6:]=0;K[6:,:6]=0;return K
def sse(H,ZTZ,YTZ,YTY):return max(float(YTY)-2*float(np.sum(YTZ*H))+float(np.trace(H@ZTZ@H.T)),0.)
def err(H,ZTZ,YTZ,YTY):return np.sqrt(sse(H,ZTZ,YTZ,YTY)/YTY)
def haar(rng,shape):
 a=rng.standard_normal((*shape,6,6));q,r=np.linalg.qr(a);sg=np.where(np.diagonal(r,axis1=-2,axis2=-1)>=0,1.,-1.);return q*sg[...,None,:]
def null_values(H,ZTZ,YTZ,YTY,seed,n=50000,batch=2000):
 rng=np.random.default_rng(seed);m=H[6:,:6];out=[]
 for start in range(0,n,batch):
  b=min(batch,n-start);q=haar(rng,(b,2));mr=q[:,0]@m@np.swapaxes(q[:,1],-1,-2);K=np.broadcast_to(H,(b,12,12)).copy();K[:,6:,:6]=mr;K[:,:6,6:]=np.swapaxes(mr,-1,-2)
  vals=np.sqrt(np.maximum(YTY-2*np.einsum("ij,bij->b",YTZ,K)+np.einsum("bij,jk,bik->b",K,ZTZ,K),0.)/YTY);out.append(vals)
 return np.concatenate(out)
def rank(H,ZTZ,YTZ,YTY,seed):
 vals=null_values(H,ZTZ,YTZ,YTY,seed);true=err(H,ZTZ,YTZ,YTY);return (1+int(np.count_nonzero(vals<=true)))/50001,float(vals.min())
def pooled_rank(items,seed):
 rng=np.random.default_rng(seed);den=sum(x[3] for x in items);true=np.sqrt(sum(sse(*x) for x in items)/den);count=0
 for start in range(0,50000,1000):
  b=min(1000,50000-start);qs=haar(rng,(b,len(items),2));num=np.zeros(b)
  for j,(H,ZTZ,YTZ,YTY) in enumerate(items):
   mr=qs[:,j,0]@H[6:,:6]@np.swapaxes(qs[:,j,1],-1,-2);K=np.broadcast_to(H,(b,12,12)).copy();K[:,6:,:6]=mr;K[:,:6,6:]=np.swapaxes(mr,-1,-2)
   num+=np.maximum(YTY-2*np.einsum("ij,bij->b",YTZ,K)+np.einsum("bij,jk,bik->b",K,ZTZ,K),0.)
  count+=int(np.count_nonzero(np.sqrt(num/den)<=true))
 return (1+count)/50001

H0=np.loadtxt(SOURCE/"H_A_SYM.tsv",delimiter="\t")
with (SOURCE/"B_CONTROL_SPACE_RAW.tsv").open(newline="") as f:rows=list(csv.DictReader(f,delimiter="\t"))
Z0=np.array([[float(r[f"state_{j}"]) for j in range(12)] for r in rows]);Y0=np.array([[float(r[f"response_{j}"]) for j in range(12)] for r in rows])
def ztz_for_amp(a):
 m=np.isclose(np.linalg.norm(Z0,axis=1),a,atol=1e-12,rtol=0);return Z0[m].T@Z0[m]
m=np.isclose(np.linalg.norm(Z0,axis=1),.02,atol=1e-12,rtol=0);Z=Z0[m];Y=Y0[m];central=Z.T@Z;YTZ=Y.T@Z;YTY=float(np.sum(Y*Y))
e=err(H0,central,YTZ,YTY);eb=err(bd(H0),central,YTZ,YTY);r,mn=rank(H0,central,YTZ,YTY,2026110901)
assert np.isclose([e,eb,mn],[.10345120235788188,.5867084710549377,.48558061628799104],atol=2e-12).all() and r==1/50001
print("orientation",e,eb,r)

items=[];E=[];B=[];T=[]
for i,seed in zip(range(1,5),range(2026110921,2026110925)):
 d=load(ROOT/"basis_resampling"/"reference"/f"basis_{i:02}.json");H=np.array(d["H"]);ZTZ=np.array(d.get("ZTZ",central));YTZ=np.array(d["YTZ"]);YTY=float(d["YTY"]);ef=err(H,ZTZ,YTZ,YTY);ed=err(bd(H),ZTZ,YTZ,YTY);rr,_=rank(H,ZTZ,YTZ,YTY,seed);assert rr==1/50001;items.append((H,ZTZ,YTZ,YTY));E.append(sse(H,ZTZ,YTZ,YTY));B.append(sse(bd(H),ZTZ,YTZ,YTY));T.append(YTY);print("basis",i,ef,ed,ef/ed,rr)
ep=np.sqrt(sum(E)/sum(T));bp=np.sqrt(sum(B)/sum(T));pr=pooled_rank(items,2026110930);assert np.isclose([ep,bp,ep/bp],[.10306333390160856,.5496413337663377,.1875101590256722],atol=2e-12).all() and pr==1/50001;print("basis pooled",ep,bp,ep/bp,pr)

items=[];E=[];B=[];T=[]
for i,seed in zip(range(1,5),range(2026110941,2026110945)):
 d=load(ROOT/"document_resampling"/"reference"/f"split_{i:02}.json");H=np.array(d["H"]);ZTZ=np.array(d.get("ZTZ",central));YTZ=np.array(d["YTZ"]);YTY=float(d["YTY"]);ef=err(H,ZTZ,YTZ,YTY);ed=err(bd(H),ZTZ,YTZ,YTY);rr,_=rank(H,ZTZ,YTZ,YTY,seed);assert rr==1/50001;items.append((H,ZTZ,YTZ,YTY));E.append(sse(H,ZTZ,YTZ,YTY));B.append(sse(bd(H),ZTZ,YTZ,YTY));T.append(YTY);print("assignment",i,ef,ed,ef/ed,rr)
ep=np.sqrt(sum(E)/sum(T));bp=np.sqrt(sum(B)/sum(T));pr=pooled_rank(items,2026110950);assert np.isclose([ep,bp,ep/bp],[.1476619621596348,.5985801253827978,.24668704472137887],atol=2e-12).all() and pr==1/50001;print("assignment pooled",ep,bp,ep/bp,pr)

f=load(ROOT/"fixed_membership"/"reference"/"fixed_membership.json");Hf=np.array(f["H_fixed"]);assert np.isclose(np.linalg.norm(Hf-H0)/np.linalg.norm(H0),1.14609699e-8,rtol=1e-6,atol=1e-14)
for a,ef0,r0,ds0 in [(.01,.09017812277517027,.1529224541097587,.05301610193101585),(.02,.09213390687124194,.15617134157787763,.04157770526819794),(.04,.09932670816042352,.16808072162556753,.029837732674455313)]:
 d=f["amplitudes"][str(a)];ZTZ=ztz_for_amp(a);YTZ=np.array(d["YTZ"]);YTY=float(d["YTY"]);ef=err(H0,ZTZ,YTZ,YTY);ed=err(bd(H0),ZTZ,YTZ,YTY);ds=np.sqrt(float(d["switch_sq"])/float(d["native_YTY"]));assert np.isclose([ef,ef/ed,ds],[ef0,r0,ds0],atol=2e-12).all();print("fixed",a,ef,ed,ef/ed,ds)
d=f["amplitudes"]["0.02"];rr,_=rank(H0,ztz_for_amp(.02),np.array(d["YTZ"]),float(d["YTY"]),2026110961);assert rr==1/50001
print("reference-output verification PASS")
