"""Native L15 radial controls. No Bank-B access is permitted before the production seal."""
import csv,gc,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from preflight import R,W,P,M,put,sha
from transformers import AutoModelForCausalLM
DEVICE='mps';DTYPE=torch.float32;BATCH=4
C_np=np.load(R/'T0_CONTRAST_BASIS.npy');D=np.load(R/'T0_DIRECTIONS.npy')
C=torch.tensor(C_np,dtype=DTYPE,device=DEVICE)
C64=torch.tensor(C_np,dtype=torch.float64)
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def records(bank):
 rows=list(csv.DictReader((R/'T0_BANK_SPLIT.tsv').open(),delimiter='\t'))
 raw=[json.loads(x) for x in (R/'inputs/Formal_Bank_T_v1.jsonl').read_text().splitlines()]
 return [(int(r['record_index']),raw[int(r['record_index'])]) for r in rows if r['bank']==bank]
def load_model():
 log('Loading exact local model FP32 / MPS / eager attention')
 m=AutoModelForCausalLM.from_pretrained(str(M),local_files_only=True,dtype=DTYPE,attn_implementation='eager').eval().to(DEVICE);m.requires_grad_(False)
 assert len(m.model.layers)==16 and len(m.model.layers[15].mlp.experts)==64
 assert m.model.layers[15].mlp.gate.bias is None
 return m

def capture(m,ids):
 cap={};layer=m.model.layers[15]
 hs=[layer.mlp.register_forward_pre_hook(lambda mod,args:cap.update(h=args[0].detach().clone())),layer.post_attention_layernorm.register_forward_pre_hook(lambda mod,args:cap.update(residual=args[0].detach().clone()))]
 try:
  with torch.no_grad():out=m(ids,labels=ids,output_router_logits=True,use_cache=False,return_dict=True)
 finally:
  for h in hs:h.remove()
 logits=torch.stack([a.detach().cpu() for a in out.router_logits])
 cap['logits0']=out.logits.detach();cap['native_loss']=float(out.loss);cap['native_aux']=float(out.aux_loss)
 # Preserve native float32 MPS softmax/top-k supports before high-precision aggregation.
 native_probs=[F.softmax(a.detach(),dim=1,dtype=torch.float) for a in out.router_logits[:-1]]
 cap['prefix_prob']=sum(p.cpu().double().sum(0) for p in native_probs)
 cap['prefix_count']=sum(F.one_hot(p.topk(8,dim=-1).indices.cpu(),64).sum((0,1)).double() for p in native_probs)
 cap['gate_logits']=out.router_logits[-1].detach().reshape(-1,64)
 return cap

def states_for_union():
 zs=[np.zeros(12)]
 for a in [.01,.02,.04]:
  for d in D:
   for s in [-1,1]:zs.append(s*a*d)
 for h in [.005,.01,.02]:
  for j in range(12):
   for s in [-1,1]:z=np.zeros(12);z[j]=s*h;zs.append(z)
 return np.array(zs)

def make_chunk(m,cap,ids,doc_indices,domains):
 h=cap['h'].reshape(-1,2048);base=cap['gate_logits'];zs=torch.tensor(states_for_union(),dtype=DTYPE,device=DEVICE)
 with torch.no_grad():
  beta=1+zs[:,6:]@C.T
  sel=(base[None,:,:]*beta[:,None,:]).topk(8,dim=-1).indices
  mask=F.one_hot(sel,64).sum(2).bool().any(0);count=mask.sum(1);k=int(count.max())
  union=torch.arange(64,device=DEVICE)[None,:].expand(h.shape[0],-1).masked_fill(~mask,64).sort(1).values[:,:k]
  values=torch.zeros((h.shape[0],k,2048),dtype=DTYPE,device=DEVICE)
  for e,expert in enumerate(m.model.layers[15].mlp.experts):
   token,slot=torch.where(union==e)
   if token.numel():values[token,slot]=expert(h[token])
 return {'v':values.cpu(),'union':union.cpu(),'residual':cap['residual'].cpu(),'base_logits':base.cpu(),'ids':ids.cpu(),'indices':doc_indices,'domains':domains,'prefix_prob':cap['prefix_prob'],'prefix_count':cap['prefix_count']}

def move(chunk,device=DEVICE,dtype=DTYPE):
 return {k:(v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) if isinstance(v,torch.Tensor) and k not in ['prefix_prob','prefix_count'] else v) for k,v in chunk.items()}

def suffix(z,ch,head,norm):
 CC=C if z.device.type=='mps' else C64
 alpha=1+CC@z[:6];beta=1+CC@z[6:]
 logits=ch['base_logits']*beta
 probs=logits.softmax(-1);weights,sel=probs.topk(8,-1)
 slot=torch.searchsorted(ch['union'].contiguous(),sel.contiguous())
 # This assertion detects any support missed by the deterministic precomputed union.
 assert bool(torch.all(slot<ch['union'].shape[1])) and bool(torch.all(ch['union'].gather(1,slot)==sel))
 v=ch['v'].gather(1,slot[:,:,None].expand(-1,-1,2048))
 y=(v*alpha[sel,None]*weights[:,:,None]).sum(1).reshape(ch['residual'].shape)+ch['residual']
 y=norm(y);out=head(y)
 loss=F.cross_entropy(out[:,:-1,:].reshape(-1,out.shape[-1]),ch['ids'][:,1:].reshape(-1))
 return loss,logits,sel,out

def direct(z,ch,m):
 mlp=m.model.layers[15].mlp;h=ch['h'];alpha=1+C@z[:6];beta=1+C@z[6:]
 overrides={'gate.weight':mlp.gate.weight*beta[:,None]}
 for e,expert in enumerate(mlp.experts):overrides[f'experts.{e}.down_proj.weight']=expert.down_proj.weight*alpha[e]
 y,logits=torch.func.functional_call(mlp,overrides,(h,))
 out=m.lm_head(m.model.norm(y+ch['residual']))
 loss=F.cross_entropy(out[:,:-1,:].reshape(-1,out.shape[-1]),ch['ids'][:,1:].reshape(-1))
 return loss,logits,out

def aux(z,chunks,hessian=False):
 # Full-bank native load balancing: counts and mean probabilities pooled over all 16 layers and every S96 token.
 b=torch.cat([c['base_logits'] for c in chunks]).double();N=b.shape[0]*16
 prefix_prob=sum(c['prefix_prob'] for c in chunks);prefix_count=sum(c['prefix_count'] for c in chunks)
 zz=torch.tensor(z,dtype=torch.float64,requires_grad=True);logits=b*(1+C64@zz[6:]);p=logits.softmax(-1);top=p.topk(8,-1).indices
 # Assignment counts follow the production float32 GPU evaluator, not float64 rankings.
 with torch.no_grad():
  u32=torch.tensor(np.asarray(z)[6:],dtype=DTYPE,device=DEVICE)
  actual=torch.cat([(c['base_logits'].to(device=DEVICE,dtype=DTYPE)*(1+C@u32)).softmax(-1).topk(8,-1).indices.cpu() for c in chunks])
 count=prefix_count+F.one_hot(actual,64).sum((0,1)).double()
 loss=.001*64*((count/N)*((prefix_prob+p.sum(0))/N)).sum()
 g=torch.autograd.grad(loss,zz,create_graph=hessian)[0]
 H=torch.stack([torch.autograd.grad(g[j],zz,retain_graph=True)[0] for j in range(12)]) if hessian else None
 return float(loss.detach()),g.detach().numpy(),None if H is None else H.detach().numpy()

def certify(m,chunk,cap):
 head=m.lm_head;norm=m.model.norm;ch=move(chunk);direct_ch={'h':cap['h'],'residual':cap['residual'],'ids':ch['ids']}
 zero=torch.zeros(12,device=DEVICE,requires_grad=True);loss,lg,sel,out=suffix(zero,ch,head,norm)
 baseline_error=float((out-cap['logits0']).abs().max());scale=float(cap['logits0'].abs().max());tests=[]
 for a in [.0,.01,.02,.04]:
  z=torch.tensor(a*D[0],dtype=DTYPE,device=DEVICE,requires_grad=True)
  ll,_,_,oo=suffix(z,ch,head,norm);gg=torch.autograd.grad(ll,z)[0]
  dd,_,do=direct(z,direct_ch,m);gd=torch.autograd.grad(dd,z)[0]
  tests.append({'amplitude':a,'max_logits_absolute_error':float((oo-do).abs().max()),'loss_absolute_error':float(abs(ll-dd)),'gradient_relative_error':float((gg-gd).norm()/gd.norm().clamp_min(1e-30))})
  del ll,oo,dd,do,gg,gd
 aux0,ag,_=aux(np.zeros(12),[chunk]);total=float(loss.detach())+aux0
 report={'status':'PASS' if baseline_error<=2e-4 and abs(total-cap['native_loss'])<=2e-5 and all(t['max_logits_absolute_error']<=3e-4 and t['gradient_relative_error']<=.005 for t in tests) else 'FAIL','dtype':'float32','device':DEVICE,'zero_control_max_logits_absolute_error':baseline_error,'baseline_logits_max_abs':scale,'configured_total_loss_optimized':total,'configured_total_loss_native':cap['native_loss'],'total_loss_absolute_error':abs(total-cap['native_loss']),'native_aux_loss':cap['native_aux'],'tolerances':{'zero_logits_absolute':2e-4,'nonzero_logits_absolute':3e-4,'total_loss_absolute':2e-5,'gradient_relative':.005},'direct_materialized_weight_scaling_vs_gain_tests':tests,'sample_indices':chunk['indices']}
 put('T1_ZERO_CONTROL_EQUIVALENCE.json',report)
 log('Equivalence: '+json.dumps(report))
 assert report['status']=='PASS','Zero/nonzero radial equivalence certification failed'
 # Cross-device precision check of local derivatives using the same Bank-A sample.
 import copy
 head64=copy.deepcopy(head).cpu().double();norm64=copy.deepcopy(norm).cpu().double();ch64=move(chunk,'cpu',torch.float64)
 z64=torch.zeros(12,dtype=torch.float64,requires_grad=True);l64,*_=suffix(z64,ch64,head64,norm64);g64=torch.autograd.grad(l64,z64)[0]
 gf=torch.autograd.grad(loss,zero)[0].detach().cpu().double()
 precision={'gradient_MPS_float32_vs_CPU_float64_relative_error':float((gf-g64).norm()/g64.norm()),'gradient_cpu64':g64.tolist(),'gradient_mps32':gf.tolist(),'limit':.005}
 put('audit/T1_PRECISION_CHECK.json',precision);assert precision['gradient_MPS_float32_vs_CPU_float64_relative_error']<=.005
 del head64,norm64,ch64,l64,g64,loss,out;gc.collect();torch.mps.empty_cache()


def collect(bank,technical=False):
 if bank=='B':
  seal=json.loads((R/'T0_PROTOCOL_SEAL.json').read_text());assert seal['production_freeze_certified'];assert all(sha(R/n)==h for n,h in seal['files'].items()),'Seal integrity failed'
 torch.set_num_threads(6);m=load_model();rr=records(bank);chunks=[];cache=W/f'work/cache/{bank}';cache.mkdir(parents=True,exist_ok=True)
 # First Bank-A document alone supplies technical equivalence samples.
 if technical:
  idx,r=rr[0];ids=torch.tensor([r['token_ids'][:96]],device=DEVICE);cap=capture(m,ids);ch=make_chunk(m,cap,ids,[idx],[r['domain']]);certify(m,ch,cap);del cap,ch
 for start in range(0,len(rr),BATCH):
  batch=rr[start:start+BATCH];ids=torch.tensor([r['token_ids'][:96] for _,r in batch],device=DEVICE)
  cap=capture(m,ids);ch=make_chunk(m,cap,ids,[i for i,_ in batch],[r['domain'] for _,r in batch]);torch.save(ch,cache/f'{start//BATCH:03}.pt');chunks.append(ch)
  if start%32==0:log(f'Captured {bank}: {start+len(batch)}/{len(rr)} documents')
  del ids,cap,ch
 head=m.lm_head.cpu();norm=m.model.norm.cpu();torch.save({'head':head.state_dict(),'norm':norm.state_dict(),'eps':norm.variance_epsilon},W/'work/cache/suffix.pt')
 del m,head,norm;gc.collect();torch.mps.empty_cache();log(f'Cache {bank} complete')
 return chunks

def load_chunks(bank):return [torch.load(p,weights_only=False) for p in sorted((W/f'work/cache/{bank}').glob('*.pt'))]
def load_suffix():
 from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm
 ss=torch.load(W/'work/cache/suffix.pt',weights_only=True);head=torch.nn.Linear(2048,50304,bias=False);head.load_state_dict(ss['head']);norm=OlmoeRMSNorm(2048,eps=ss['eps']);norm.load_state_dict(ss['norm']);head.requires_grad_(False);norm.requires_grad_(False);return head.to(DEVICE),norm.to(DEVICE)

def measure(chunks,z,head,norm,hessian=False,domainwise=False):
 N=sum(len(c['indices']) for c in chunks);g=np.zeros(12);H=np.zeros((12,12));loss=0.;by={}
 for i,c in enumerate(chunks):
  ch=move(c);zz=torch.tensor(z,dtype=DTYPE,device=DEVICE,requires_grad=True);l,_,_,_=suffix(zz,ch,head,norm);gg=torch.autograd.grad(l,zz,create_graph=hessian)[0];w=len(c['indices'])/N;loss+=float(l.detach())*w;g+=gg.detach().cpu().double().numpy()*w
  if hessian:
   hh=torch.stack([torch.autograd.grad(gg[j],zz,retain_graph=True)[0] for j in range(12)]);H+=hh.detach().cpu().double().numpy()*w;del hh
  if domainwise:
   # Chunks are within one domain because each domain has 32 rows and BATCH=4.
   assert len(set(c['domains']))==1
   d=c['domains'][0];by.setdefault(d,{'chunks':[],'lm_gradient_sum':np.zeros(12),'docs':0});by[d]['chunks'].append(c);by[d]['lm_gradient_sum']+=gg.detach().cpu().double().numpy()*len(c['indices']);by[d]['docs']+=len(c['indices'])
  del l,gg,ch,zz
 al,ag,ah=aux(z,chunks,hessian);loss+=al;g+=ag
 if hessian:H+=ah
 dg={}
 if domainwise:
  for d,vv in by.items():
   # Separate full configured objective per domain, kept descriptive only.
   dg[d]=vv['lm_gradient_sum']/vv['docs']+aux(z,vv['chunks'])[1]
 return loss,g,H if hessian else None,dg

def routing(chunks,z):
 # Diagnostic supports use exactly the nonlinear evaluator's float32 MPS operations.
 with torch.no_grad():
  beta=1+C@torch.tensor(np.asarray(z)[6:],dtype=DTYPE,device=DEVICE)
  oldsets=[];newsets=[];oldvals=[];newvals=[]
  for c in chunks:
   b=c['base_logits'].to(DEVICE);logits=b*beta
   oldsets.append(b.softmax(-1).topk(8,-1).indices.sort(1).values.cpu());newsets.append(logits.softmax(-1).topk(8,-1).indices.sort(1).values.cpu())
   oldvals.append(b.topk(9,-1).values.cpu().double());newvals.append(logits.topk(9,-1).values.cpu().double())
  base=torch.cat(oldsets);sel=torch.cat(newsets);v0=torch.cat(oldvals);vn=torch.cat(newvals)
 same=(base==sel).all(1);common=(base[:,:,None]==sel[:,None,:]).sum((1,2));sym=16-2*common
 margin=v0[:,7]-v0[:,8];changed_margin=vn[:,7]-vn[:,8]
 bins=[0.,.01,.05,.1,.5,float('inf')];cond=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  take=(margin>=lo)&(margin<hi);cond.append({'lo':lo,'hi':None if hi==float('inf') else hi,'tokens':int(take.sum()),'turnover':float((~same[take]).double().mean()) if take.any() else None})
 return {'support_turnover_fraction':float((~same).double().mean()),'mean_support_symmetric_difference':float(sym.double().mean()),'baseline_margin_median':float(margin.median()),'baseline_margin_p05':float(torch.quantile(margin,.05)),'perturbed_margin_median':float(changed_margin.median()),'margin_conditioned':cond}