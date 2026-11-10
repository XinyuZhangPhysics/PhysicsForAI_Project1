import os,sys
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[2]/"source_to_heldout"/"source";sys.path.insert(0,str(BASE))
import runtime_config as rc
import engine as e
from deterministic_index import install as install_deterministic
from fixed_membership import install as install_fixed

def main():
 e.CACHE_ROOT=Path(os.environ.get("PROJECT1_NMIX_CACHE_ROOT",rc.CACHE_ROOT));torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
 install_deterministic(e);native_suffix,native_aux=install_fixed(e);head,norm=e.load_suffix();A=e.load_chunks("A");B=e.load_chunks("B");zero=np.zeros(12)
 def measure(chunks,fixed,z=zero,hessian=False):
  e.suffix=e.suffix_fixed_membership if fixed else native_suffix;e.aux=e.aux_fixed_membership if fixed else native_aux;return e.measure(chunks,z,head,norm,hessian=hessian)
 _,g0,_,_=measure(B,True);Hs=[measure(A,True,hessian=True)[2] for _ in range(2)];assert Hs[0].tobytes()==Hs[1].tobytes();H=(Hs[0]+Hs[0].T)/2
 states=np.array([sign*a*d for a in (.01,.02,.04) for d in e.D for sign in (-1,1)]);grads=[]
 for k,z in enumerate(states):
  _,g,_,_=measure(B,True,z=z);grads.append(g);print("fixed state",k+1,"/",len(states),flush=True)
 out=Path(os.environ.get("PROJECT1_NMIX_FIXED_OUTPUT",Path(__file__).resolve().parents[1]/"outputs"));out.mkdir(parents=True,exist_ok=True);np.save(out/"H_FIXED_SYM.npy",H);grads=np.asarray(grads);np.savez(out/"B_FIXED_GRADIENT_RESPONSES.npz",states=states,gradients=grads,g0=g0,responses=grads-g0)
if __name__=="__main__":main()
