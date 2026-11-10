import os,sys,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BASE=Path(__file__).resolve().parents[2]/"source_to_heldout"/"source";sys.path.insert(0,str(BASE))
import runtime_config as rc
import engine as e

class PrefixComplete(Exception):pass

def main():
 cache=Path(os.environ.get("PROJECT1_NMIX_CACHE_ROOT",rc.CACHE_ROOT));out=Path(os.environ.get("PROJECT1_NMIX_DOCUMENT_OUTPUT",Path(__file__).resolve().parents[1]/"outputs"));out.mkdir(parents=True,exist_ok=True)
 e.CACHE_ROOT=cache
 model=e.load_model();logits=[];handles=[]
 for layer in model.model.layers[:15]:handles.append(layer.mlp.register_forward_hook(lambda mod,args,result:logits.append(result[1].detach())))
 handles.append(model.model.layers[15].mlp.register_forward_pre_hook(lambda mod,args:(_ for _ in ()).throw(PrefixComplete())))
 prob=np.zeros((448,64));count=np.zeros((448,64),dtype=np.int64)
 chunks=sorted((cache/"A").glob("*.pt"))+sorted((cache/"B").glob("*.pt"))
 for k,p in enumerate(chunks):
  c=torch.load(p,weights_only=False);logits.clear();ids=c["ids"].to("mps")
  with torch.no_grad():
   try:model(ids,output_router_logits=True,use_cache=False,return_dict=True)
   except PrefixComplete:pass
  assert len(logits)==15
  ps=[F.softmax(a,dim=1,dtype=torch.float) for a in logits]
  rp=sum(x.cpu().double().reshape(4,96,64).sum(1) for x in ps)
  rcnt=sum(F.one_hot(x.topk(8,dim=-1).indices.cpu(),64).reshape(4,96,8,64).sum((1,2)) for x in ps)
  assert float((rp.sum(0)-c["prefix_prob"]).abs().max())<=1e-9
  assert torch.equal(rcnt.sum(0).double(),c["prefix_count"])
  for j,idx in enumerate(c["indices"]):prob[idx]=rp[j].numpy();count[idx]=rcnt[j].numpy()
  if (k+1)%16==0:print("prefix chunks",k+1,"/",len(chunks),flush=True)
 for h in handles:h.remove()
 del model;gc.collect();torch.mps.empty_cache()
 np.save(out/"PREFIX_PROB_BY_RECORD.npy",prob);np.save(out/"PREFIX_COUNT_BY_RECORD.npy",count)
if __name__=="__main__":main()
