"""Frozen basis-resampling execution. Derived engine retains historical suffix/aux/measure.
Only union construction, basis binding, and scheduling are changed. No H_B measured.
"""
import csv,gc,json,shutil,time,traceback,sys,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from runtime_config import *

def log(s):
 stamp=time.strftime('%Y-%m-%d %H:%M:%S');print(stamp,s,flush=True)
 with (O/'RUN_LOG.md').open('a') as f:f.write(f'- {stamp}: {s}\n')
def validate():
 cfg=CONFIG
 for n,h in cfg['model']['expected_sha256'].items():assert sha(M/n)==h,n
 assert BANK.is_file()
 assert sha(BANK)==cfg['bank']['archived_jsonl_sha256']
 assert torch.backends.mps.is_available(),'MPS device unavailable'
 log('Model, bank and runtime identities verified.')
validate()
import engine as e
from deterministic_index import install
installation=install(e)
torch.set_num_threads(6)
Cs=[generate_basis(seed) for seed in BASIS_SEEDS]
C32=[torch.tensor(c,dtype=torch.float32,device='mps') for c in Cs]
D=np.loadtxt(R/'T0_DIRECTIONS.tsv',delimiter='\\t')
STATES=np.array([s*.02*d for d in D for s in [-1,1]])
CACHE=CACHE_ROOT

def bind(b):
 e.C_np=Cs[b-1];e.C=C32[b-1];e.C64=torch.tensor(e.C_np,dtype=torch.float64)
def make_chunk(m,cap,ids,indices,domains):
 h=cap['h'].reshape(-1,2048);base=cap['gate_logits']
 with torch.no_grad():
  mask=F.one_hot(base.softmax(-1).topk(8,-1).indices,64).sum(1).bool()
  # Match runtime matrix-vector multiply, softmax, and topk, including float32 rounding.
  for c in C32:
   for z in STATES:
    beta=1+c@torch.tensor(z[6:],dtype=torch.float32,device='mps')
    sel=(base*beta).softmax(-1).topk(8,-1).indices
    mask|=F.one_hot(sel,64).sum(1).bool()
  counts=mask.sum(1);k=int(counts.max())
  union=torch.arange(64,device='mps')[None,:].expand(h.shape[0],-1).masked_fill(~mask,64).sort(1).values[:,:k]
  values=torch.zeros((h.shape[0],k,2048),dtype=torch.float32,device='mps')
  for expert_id,expert in enumerate(m.model.layers[15].mlp.experts):
   token,slot=torch.where(union==expert_id)
   if token.numel():values[token,slot]=expert(h[token])
 return dict(v=values.cpu(),union=union.cpu(),union_counts=counts.cpu(),residual=cap['residual'].cpu(),base_logits=base.cpu(),ids=ids.cpu(),indices=indices,domains=domains,prefix_prob=cap['prefix_prob'],prefix_count=cap['prefix_count'])

def certify(m,ch,cap,b):
 bind(b);outdir=f'basis_{b:02}'
 device_ch=e.move(ch);direct_ch=dict(h=cap['h'],residual=cap['residual'],ids=device_ch['ids']);tests=[]
 for i,z in enumerate([np.zeros(12),STATES[0]]):
  zz=torch.tensor(z,dtype=torch.float32,device='mps',requires_grad=True)
  loss,_,_,out=e.suffix(zz,device_ch,m.lm_head,m.model.norm)
  g=torch.autograd.grad(loss,zz)[0]
  dl,_,do=e.direct(zz,direct_ch,m);dg=torch.autograd.grad(dl,zz)[0]
  tests.append(dict(state=z.tolist(),max_logits_absolute_error=float((out-do).abs().max()),loss_absolute_error=float(abs(loss-dl)),gradient_relative_error=float((g-dg).norm()/dg.norm().clamp_min(1e-30))))
  if i==0:
   baseline_error=float((out-cap['logits0']).abs().max());aux0,_,_=e.aux(np.zeros(12),[ch]);total=float(loss.detach())+aux0
  del out,do,loss,dl,g,dg
 report=dict(status='PASS',basis=b,deterministic_gain_backend=installation,sample_indices=ch['indices'],zero_logits_error=baseline_error,total_loss_native=cap['native_loss'],total_loss_cached=total,total_loss_absolute_error=abs(total-cap['native_loss']),tests=tests,tolerances=dict(zero_logits_absolute=2e-4,nonzero_logits_absolute=3e-4,total_loss_absolute=2e-5,gradient_relative=.005))
 passed=baseline_error<=2e-4 and abs(total-cap['native_loss'])<=2e-5 and all(t['max_logits_absolute_error']<=3e-4 and t['gradient_relative_error']<=.005 for t in tests)
 report['status']='PASS' if passed else 'FAIL';put(f'{outdir}/technical_certification.json',report)
 log(f'Basis {b} technical certification {report["status"]}: '+json.dumps(tests))
 assert passed,f'Basis {b}: direct radial vs cache numerical certification failure'

def build_caches():
 m=e.load_model();log('Exact checkpoint loaded in FP32 / MPS / eager attention')
 rr=e.records('A');idx,r=rr[0];ids=torch.tensor([r['token_ids'][:96]],device='mps');cap=e.capture(m,ids);ch=make_chunk(m,cap,ids,[idx],[r['domain']])
 torch.save(ch,CACHE/'A_technical.pt')
 for b in range(1,5):certify(m,ch,cap,b)
 del ch,cap,ids;gc.collect();torch.mps.empty_cache()
 audit=dict(procedure='Baseline plus all four frozen bases at all 24 signed amplitude-0.02 states; runtime native float32 softmax/topk; exact union containment asserted by historical suffix',objective='Historical mean causal CE plus one full-bank coefficient-0.001 native auxiliary; prefix sums/counts pooled, basis-specific last-layer probabilities/counts recomputed',banks={})
 for bank in ['A','B']:
  rr=e.records(bank);folder=CACHE/bank;folder.mkdir(exist_ok=True);files=[];counts=[]
  for start in range(0,len(rr),4):
   batch=rr[start:start+4];ids=torch.tensor([r['token_ids'][:96] for _,r in batch],device='mps')
   cap=e.capture(m,ids);ch=make_chunk(m,cap,ids,[i for i,_ in batch],[r['domain'] for _,r in batch]);path=folder/f'{start//4:03}.pt';torch.save(ch,path)
   files.append(dict(file=str(path.relative_to(W)),sha256=sha(path),indices=ch['indices']));counts.extend(ch['union_counts'].tolist())
   del ids,cap,ch
   log(f'Common {bank} cache {start+len(batch)}/{len(rr)} documents')
  audit['banks'][bank]=dict(chunks=len(files),documents=len(rr),files=files,union_width=dict(mean=float(np.mean(counts)),median=float(np.median(counts)),p95=float(np.quantile(counts,.95)),max=int(max(counts)),min=int(min(counts))))
  put('CACHE_AUDIT.json',audit)
 head=m.lm_head.cpu();norm=m.model.norm.cpu();torch.save(dict(head=head.state_dict(),norm=norm.state_dict(),eps=norm.variance_epsilon),CACHE/'suffix.pt');audit['suffix_sha256']=sha(CACHE/'suffix.pt');put('CACHE_AUDIT.json',audit)
 del m,head,norm;gc.collect();torch.mps.empty_cache()

def load_chunks(bank):return [torch.load(p,weights_only=False) for p in sorted((CACHE/bank).glob('*.pt'))]
def load_suffix():
 from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
 ss=torch.load(CACHE/'suffix.pt',weights_only=True);head=torch.nn.Linear(2048,50304,bias=False);head.load_state_dict(ss['head']);norm=OlmoeRMSNorm(2048,eps=ss['eps']);norm.load_state_dict(ss['norm']);head.requires_grad_(False);norm.requires_grad_(False);return head.to('mps'),norm.to('mps')

def measure(chunks,z,head,norm,hessian=False):
 # Historical objective and derivatives. Logging wrapper only.
 return e.measure(chunks,z,head,norm,hessian=hessian)

def neural():
 build_caches();torch.use_deterministic_algorithms(True)
 head,norm=load_suffix();chunks=load_chunks('A')
 for b in range(1,5):
  bind(b);folder=O/f'basis_{b:02}';Hs=[]
  for repeat in range(2):
   log(f'Basis {b} A Hessian pass {repeat+1}/2 starting')
   loss,g,H,_=measure(chunks,np.zeros(12),head,norm,hessian=True);Hs.append(H)
   np.save(folder/('H_A_raw.npy' if repeat==0 else 'H_A_repeat.npy'),H)
   if repeat==0:np.savez(folder/'A_baseline.npz',gradient=g,loss=loss)
   log(f'Basis {b} A Hessian pass {repeat+1}/2 complete')
  defect=float(np.linalg.norm(Hs[0]-Hs[0].T)/np.linalg.norm(Hs[0]));equal=np.array_equal(Hs[0],Hs[1]);finite=bool(np.isfinite(Hs).all())
  put(f'basis_{b:02}/hessian_certification.json',dict(repeat_bitwise_identical=equal,raw_symmetry_relative_frobenius=defect,symmetry_limit=.05,finite=finite,backend=installation))
  assert equal and finite and defect<=.05,f'Basis {b} deterministic Hessian certification failure'
  np.save(folder/'H_A_sym.npy',(Hs[0]+Hs[0].T)/2)
 del chunks;gc.collect();chunks=load_chunks('B')
 for b in range(1,5):
  bind(b);folder=O/f'basis_{b:02}';log(f'Basis {b} B baseline gradient starting');loss0,g0,_,_=measure(chunks,np.zeros(12),head,norm);gs=[];losses=[];turn=[]
  for j,z in enumerate(STATES):
   loss,g,_,_=measure(chunks,z,head,norm);rt=e.routing(chunks,z)
   gs.append(g);losses.append(loss);turn.append(dict(state_id=j,direction_id=j//2,sign=[-1,1][j%2],amplitude=.02,**rt))
   np.savez(folder/'B_raw_gradients_responses.npz',baseline_gradient=g0,baseline_loss=loss0,gradients=np.array(gs),responses=np.array(gs)-g0,states=STATES[:len(gs)],losses=losses)
   log(f'Basis {b} B state {j+1}/24 complete')
  put(f'basis_{b:02}/routing_diagnostics.json',turn)
  with (folder/'support_turnover.csv').open('w') as f:
   names=[k for k in turn[0] if k!='margin_conditioned'];writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows({k:t[k] for k in names} for t in turn)
 # Seal before any outcome analysis or geometric null computation.
 files={str(f.relative_to(O)):sha(f) for f in sorted(O.rglob('*')) if f.is_file() and f.name!='NN_OUTPUT_SEAL.json'}
 put('NN_OUTPUT_SEAL.json',dict(status='SEALED_NEURAL_OUTPUTS_BEFORE_GEOMETRIC_NULL',files=files,created=time.strftime('%Y-%m-%d %H:%M:%S')))
 log('All four basis-resampling neural-output sets complete.')
if __name__=='__main__':
 try:neural()
 except Exception as ex:
  put('EXECUTION_ERROR.json',dict(type=type(ex).__name__,message=str(ex),traceback=traceback.format_exc()));log('Execution stopped: '+repr(ex));raise