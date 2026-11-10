import numpy as np
import torch
from runtime_config import M,BANK,O,CACHE_ROOT,CONFIG,sha,put
import engine as e
from deterministic_index import install

def validate():
 for name,digest in CONFIG["model"]["expected_sha256"].items():assert sha(M/name)==digest,name
 assert sha(BANK)==CONFIG["bank"]["archived_jsonl_sha256"]
 assert torch.backends.mps.is_available()
 print("input identities verified",flush=True)

def main():
 validate();torch.set_num_threads(6);torch.use_deterministic_algorithms(True);install(e)
 e.collect("A",technical=True);e.collect("B",technical=False)
 head,norm=e.load_suffix();A=e.load_chunks("A");B=e.load_chunks("B")
 Hs=[]
 for _ in range(2):
  _,_,H,_=e.measure(A,np.zeros(12),head,norm,hessian=True);Hs.append(H)
 assert Hs[0].tobytes()==Hs[1].tobytes()
 H=(Hs[0]+Hs[0].T)/2
 np.save(O/"H_A_RAW.npy",Hs[0]);np.save(O/"H_A_REPEAT.npy",Hs[1]);np.save(O/"H_A_SYM.npy",H)
 loss0,g0,_,_=e.measure(B,np.zeros(12),head,norm)
 states=np.array([sign*a*d for a in (.01,.02,.04) for d in e.D for sign in (-1,1)])
 grads=[];losses=[]
 for k,z in enumerate(states):
  loss,g,_,_=e.measure(B,z,head,norm);grads.append(g);losses.append(loss);print("heldout state",k+1,"/",len(states),flush=True)
 grads=np.asarray(grads)
 np.savez(O/"B_GRADIENT_RESPONSES.npz",states=states,gradients=grads,g0=g0,responses=grads-g0,losses=np.asarray(losses),baseline_loss=loss0)
 manifest={}
 for p in sorted(CACHE_ROOT.rglob("*")):
  if p.is_file():manifest[str(p.relative_to(CACHE_ROOT))]=sha(p)
 put("CACHE_MANIFEST.json",{"files":manifest})
 print("source-to-heldout measurement complete",flush=True)
if __name__=="__main__":main()
