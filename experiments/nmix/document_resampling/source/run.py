import os,sys,csv,gc
from pathlib import Path
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[2]/"source_to_heldout"/"source";sys.path.insert(0,str(BASE))
import runtime_config as rc
import engine as e
from deterministic_index import install

SEEDS=(2026110911,2026110912,2026110913,2026110914)

def assignments(rows,seed):
 rng=np.random.default_rng(seed);out={}
 for dom in sorted({r["domain"] for r in rows}):
  ids=sorted(int(r["record_index"]) for r in rows if r["domain"]==dom);p=rng.permutation(ids);A=set(map(int,p[:32]))
  for idx in ids:out[idx]="A" if idx in A else "B"
 return out

def main():
 cache=Path(os.environ.get("PROJECT1_NMIX_CACHE_ROOT",rc.CACHE_ROOT));out=Path(os.environ.get("PROJECT1_NMIX_DOCUMENT_OUTPUT",Path(__file__).resolve().parents[1]/"outputs"));out.mkdir(parents=True,exist_ok=True)
 e.CACHE_ROOT=cache;torch.set_num_threads(6);torch.use_deterministic_algorithms(True);install(e)
 prob=np.load(out/"PREFIX_PROB_BY_RECORD.npy");count=np.load(out/"PREFIX_COUNT_BY_RECORD.npy")
 allchunks=[torch.load(p,weights_only=False) for p in sorted((cache/"A").glob("*.pt"))+sorted((cache/"B").glob("*.pt"))];lookup={}
 for c in allchunks:
  for j,idx in enumerate(c["indices"]):lookup[int(idx)]=(c,j)
 rows=list(csv.DictReader((rc.R/"T0_BANK_SPLIT.tsv").open(),delimiter="\t"));assert len(lookup)==448
 def bank_chunks(bankmap,bank):
  selected=sorted([idx for idx,b in bankmap.items() if b==bank],key=lambda idx:(next(r["domain"] for r in rows if int(r["record_index"])==idx),idx));res=[]
  for start in range(0,224,4):
   ids=selected[start:start+4];src=[lookup[i] for i in ids];K=max(c["union"].shape[1] for c,j in src);vs=[];us=[];rs=[];bs=[];tok=[];domains=[]
   for c,j in src:
    v=c["v"][j*96:(j+1)*96];u=c["union"][j*96:(j+1)*96];pad=K-u.shape[1]
    if pad:v=torch.cat([v,torch.zeros((96,pad,2048),dtype=v.dtype)],1);u=torch.cat([u,torch.full((96,pad),64,dtype=u.dtype)],1)
    vs.append(v);us.append(u);rs.append(c["residual"][j:j+1]);bs.append(c["base_logits"][j*96:(j+1)*96]);tok.append(c["ids"][j:j+1]);domains.append(c["domains"][j])
   res.append({"v":torch.cat(vs),"union":torch.cat(us),"residual":torch.cat(rs),"base_logits":torch.cat(bs),"ids":torch.cat(tok),"indices":ids,"domains":domains,"prefix_prob":torch.tensor(prob[ids].sum(0)),"prefix_count":torch.tensor(count[ids].sum(0),dtype=torch.float64)})
  return res
 head,norm=e.load_suffix();states=np.array([s*.02*d for d in e.D for s in (-1,1)])
 for si,seed in enumerate(SEEDS,1):
  m=assignments(rows,seed);A=bank_chunks(m,"A");B=bank_chunks(m,"B");Hs=[e.measure(A,np.zeros(12),head,norm,hessian=True)[2] for _ in range(2)];assert Hs[0].tobytes()==Hs[1].tobytes();H=(Hs[0]+Hs[0].T)/2
  _,g0,_,_=e.measure(B,np.zeros(12),head,norm);grads=[]
  for k,z in enumerate(states):
   _,g,_,_=e.measure(B,z,head,norm);grads.append(g)
  dest=out/f"split_{si:02}";dest.mkdir(exist_ok=True);np.save(dest/"H_A_SYM.npy",H);grads=np.asarray(grads);np.savez(dest/"B_RESPONSES.npz",states=states,gradients=grads,g0=g0,responses=grads-g0)
  print("split",si,"complete",flush=True);del A,B;gc.collect();torch.mps.empty_cache()
if __name__=="__main__":main()
