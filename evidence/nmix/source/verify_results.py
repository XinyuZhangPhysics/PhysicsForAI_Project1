"""Independent deterministic audit of saved NMIX scalars, bootstrap and routing supports."""
import csv,json,hashlib
import numpy as np
from preflight import R,sha

def main():
 checks={}
 for sn in ['T0_PROTOCOL_SEAL.json','audit/PRIMARY_RESULTS_SEAL.json']:
  seal=json.loads((R/sn).read_text());assert all(sha(R/n)==h for n,h in seal['files'].items());checks[sn]='PASS'
 cfg=json.loads((R/'T0_FROZEN_CONFIG.json').read_text());assert cfg==json.loads((R/'frozen_package/protocol/nmix_frozen_protocol.json').read_text())
 H=np.load(R/'T2_HESSIAN_A_SYM.npy');raw=np.load(R/'T2_HESSIAN_A_RAW.npy');assert np.allclose(H,(raw+raw.T)/2,atol=0,rtol=0);Hd=H.copy();Hd[:6,6:]=0;Hd[6:,:6]=0
 rr=list(csv.DictReader((R/'T3_BANK_B_STATE_METRICS.csv').open()));assert len(rr)==72
 arrays=np.load(R/'audit/B_raw_gradients_responses.npz');responses=arrays['responses'];states=arrays['states'];assert np.allclose(arrays['gradients']-arrays['g0'],responses,atol=0,rtol=0)
 for i,row in enumerate(rr):
  r=responses[i];z=states[i];ef=np.linalg.norm(r-H@z);ed=np.linalg.norm(r-Hd@z)
  assert np.isclose(ef,float(row['full_error']),rtol=1e-13);assert np.isclose(ed,float(row['no_mixed_error']),rtol=1e-13)
  assert np.isclose(np.linalg.norm(r),float(row['response_norm']),rtol=1e-13)
  co=lambda a,b:np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)
  assert np.isclose(co(r,H@z),float(row['full_cosine']),rtol=1e-13)
 summaries=json.loads((R/'T3_BANK_B_PRIMARY_SUMMARY.json').read_text());p=summaries['primary']
 ids=[i for i,row in enumerate(rr) if float(row['amplitude'])==.02];numer=sum(float(rr[i]['full_error'])**2 for i in ids);den=sum(float(rr[i]['no_mixed_error'])**2 for i in ids);assert np.isclose(np.sqrt(numer/den),p['R'],rtol=1e-13)
 bootstrap=np.loadtxt(R/'T3_DIRECTION_BOOTSTRAP.csv',delimiter=',',skiprows=1);assert bootstrap.shape==(10000,3);rng=np.random.Generator(np.random.PCG64(2026103003))
 for k in range(10000):
  chosen=[ids[2*i+s] for i in rng.integers(0,12,12) for s in range(2)]
  ratio=np.sqrt(sum(float(rr[i]['full_error'])**2 for i in chosen)/sum(float(rr[i]['no_mixed_error'])**2 for i in chosen));median=np.median([float(rr[i]['full_cosine']) for i in chosen]);assert np.allclose(bootstrap[k],[k,ratio,median],rtol=1e-13,atol=1e-15)
 assert np.allclose(np.quantile(bootstrap[:,1],[.025,.975]),p['R_bootstrap_ci95'],rtol=1e-13)
 checks['primary_metrics_and_all_10000_bootstrap_replicates']='PASS'
 split=list(csv.DictReader((R/'T0_BANK_SPLIT.tsv').open(),delimiter='\t'));assert sum(r['bank']=='A' for r in split)==sum(r['bank']=='B' for r in split)==224
 with (R/'T1_ROUTING_BASELINE_DIAGNOSTICS.csv').open() as f:baseline=list(csv.DictReader(f))
 assert len(baseline)==224*96
 # Match diagnostics to native float32 top-k, evaluated per identical four-document chunk.
 import torch
 C=torch.tensor(np.load(R/'T0_CONTRAST_BASIS.npy'),dtype=torch.float32,device='mps');b=np.load(R/'audit/B_baseline_router_logits.npy');n=4*96
 old=[]
 for start in range(0,len(b),n):
  t=torch.tensor(b[start:start+n],device='mps');p0=t.softmax(-1);support=p0.topk(8,-1).indices.cpu();old.append(support.sort(-1).values)
  for ii in range(len(t)):
   actual=support[ii].tolist();saved=json.loads(baseline[start+ii]['top8_experts']);assert actual==saved
 base=torch.cat(old)
 for k,z in enumerate(states):
  beta=1+C@torch.tensor(z[6:],dtype=torch.float32,device='mps');new=[]
  for start in range(0,len(b),n):
   t=torch.tensor(b[start:start+n],device='mps');new.append((t*beta).softmax(-1).topk(8,-1).indices.sort(-1).values.cpu())
  selected=torch.cat(new);changed=(base!=selected).any(-1);common=(base[:,:,None]==selected[:,None,:]).sum((1,2));sym=16-2*common
  assert abs(float(changed.double().mean())-float(rr[k]['support_turnover_fraction']))<1e-15
  assert abs(float(sym.double().mean())-float(rr[k]['mean_support_symmetric_difference']))<1e-15
 checks['all_72_native_routing_diagnostics_and_baseline_supports']='PASS'
 checks['frozen_config_and_balanced_split']='PASS';checks['hessian_symmetrization']='PASS'
 (R/'audit/FINAL_VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()