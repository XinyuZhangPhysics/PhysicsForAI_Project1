"""Revision-4 NMIX staged execution. The Bank-B gate verifies an immutable pre-primary code seal."""
import argparse,copy,csv,gc,json,time
from pathlib import Path
import numpy as np
import torch
from engine import *

def hessian_sample_reference(ch,head,norm):
 zz=torch.zeros(12,dtype=DTYPE,device=DEVICE,requires_grad=True);l,*_=suffix(zz,move(ch),head,norm);g=torch.autograd.grad(l,zz,create_graph=True)[0];hh=torch.stack([torch.autograd.grad(g[j],zz,retain_graph=True)[0] for j in range(12)]).detach().cpu().double().numpy()
 h64=copy.deepcopy(head).cpu().double();n64=copy.deepcopy(norm).cpu().double();z64=torch.zeros(12,dtype=torch.float64,requires_grad=True);ll,*_=suffix(z64,move(ch,'cpu',torch.float64),h64,n64);gg=torch.autograd.grad(ll,z64,create_graph=True)[0];href=torch.stack([torch.autograd.grad(gg[j],z64,retain_graph=True)[0] for j in range(12)]).detach().numpy()
 result={'H_MPS':hh.tolist(),'H_CPU_reference':href.tolist(),'relative_frobenius_discrepancy':float(np.linalg.norm(hh-href)/np.linalg.norm(href)),'sample_record_indices':ch['indices'],'reference':'CPU float64 controls/cache/head with native RMSNorm float32 variance computation','limit':.10}
 put('audit/AD_BACKEND_REFERENCE.json',result);del h64,n64,ll,gg,l,g;gc.collect();torch.mps.empty_cache();return result

def seal():
 refs=Path('<LOCAL_HOME>/AI_Workspace/Projects/Codex/2026-10-16/jie/work/public-code/experiments/nativeolmoe/native_control_dependencies/project1_r2')
 (R/'technical_reference').mkdir(exist_ok=True)
 import shutil,transformers.models.olmoe.modeling_olmoe as mo
 for p in [refs/'model_io.py',refs/'__init__.py',Path(mo.__file__)]:shutil.copyfile(p,R/'technical_reference'/p.name)
 (R/'T0_SOURCE_SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted((R/'source').glob('*.py')))+'\n')
 filenames=['T0_FROZEN_CONFIG.json','T0_CONTRAST_BASIS.npy','T0_DIRECTIONS.npy','T0_BANK_SPLIT.tsv','T0_MODEL_AUDIT.json','T0_BANK_AUDIT.json','ENVIRONMENT.txt','T1_ZERO_CONTROL_EQUIVALENCE.json','T1_HESSIAN_BACKEND_CERTIFICATION.json','T1_LAYER_ARCHITECTURE_AUDIT.md','T1_OBJECTIVE_AUDIT.md','T2_HESSIAN_A_RAW.npy','T2_HESSIAN_A_SYM.npy','T0_SOURCE_SHA256SUMS.txt']
 files=[R/f for f in filenames]+list((R/'source').glob('*.py'))+[p for p in (R/'frozen_package').rglob('*') if p.is_file()]+list((R/'technical_reference').glob('*.py'))+list((R/'audit').glob('*REFERENCE.json'))
 put('T0_PROTOCOL_SEAL.json',{'protocol_id':'Project1_NMIX_v1','run_revision':4,'production_freeze_certified':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'bank_B_scientific_evaluated':False,'technical_repair':'Revision 2 diagnostic float64 support ranking differed on one token-state; revision 3 uses evaluator float32 MPS softmax/top-k for support diagnostics and native auxiliary assignment counts. All numerical caches and scientific outputs regenerated from inputs.','backend':'exact second-order AD','dtype':'float32 MPS LM suffix + float64 CPU bank-aggregate auxiliary','environment':json.loads((R/'audit/ENVIRONMENT.json').read_text()),'files':{str(p.relative_to(R)):sha(p) for p in files}})
 log('PRE-PRIMARY CODE/PROTOCOL SEAL WRITTEN; Bank B may now be evaluated.')

def preprimary():
 chunks=load_chunks('A');assert len(chunks)==56
 head,norm=load_suffix();ref=None;aderr=None
 try:
  ref=hessian_sample_reference(chunks[0],head,norm);log('AD reference discrepancy '+str(ref['relative_frobenius_discrepancy']))
  assert ref['relative_frobenius_discrepancy']<=.10,'AD device-reference numerical mismatch'
  log('Measuring exact Bank-A Hessian')
  loss,g,H,_=measure(chunks,np.zeros(12),head,norm,hessian=True)
  symmetry=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H),1e-30));assert symmetry<=.05,'AD Hessian symmetry failure'
  report={'status':'PASS','backend':'second_order_autodiff','raw_symmetry_defect':symmetry,'raw_symmetry_defect_limit':.05,'reference':ref['relative_frobenius_discrepancy'],'reference_limit':.10,'finite_difference_used':False,'loss_A0':loss,'gradient_A0':g.tolist()}
 except Exception as e:
  aderr=repr(e);log('AD backend unavailable/uncertified: '+aderr)
  matrices={}
  for h in [.005,.01,.02]:
   cols=[]
   for j in range(12):
    z=np.zeros(12);z[j]=h;gp=measure(chunks,z,head,norm)[1];gm=measure(chunks,-z,head,norm)[1];cols.append((gp-gm)/(2*h))
   matrices[h]=np.stack(cols,axis=1);np.save(R/f'audit/H_FD_{h}.npy',matrices[h]);log(f'FD Hessian at h={h} complete')
  candidates=[];selected=None
  for i,h in enumerate([.005,.01,.02]):
   hh=matrices[h];sym=float(np.linalg.norm(hh-hh.T)/max(np.linalg.norm(hh),1e-30));larger=matrices[[.005,.01,.02][i+1]] if i<2 else None
   stable=None if larger is None else float(np.linalg.norm(hh-larger)/max(np.linalg.norm(larger),1e-30));passed=sym<=.05 and stable is not None and stable<=.10
   candidates.append({'h':h,'symmetry_defect':sym,'step_stability_relative_frobenius':stable,'pass':passed})
   if selected is None and passed:selected=h
  report={'status':'PASS' if selected is not None else 'FAIL','backend':'central_finite_difference_of_control_gradient','exact_AD_failure':aderr,'candidates':candidates,'selected_step':selected,'symmetry_defect_limit':.05,'stability_limit':.10}
  put('T1_HESSIAN_BACKEND_CERTIFICATION.json',report)
  assert selected is not None,'No frozen Hessian backend passes certification'
  H=matrices[selected];loss,g,_,_=measure(chunks,np.zeros(12),head,norm)
 put('T1_HESSIAN_BACKEND_CERTIFICATION.json',report)
 np.save(R/'T2_HESSIAN_A_RAW.npy',H);Hsym=(H+H.T)/2;np.save(R/'T2_HESSIAN_A_SYM.npy',Hsym);np.save(R/'audit/g_A0.npy',g)
 put('T2_HESSIAN_BLOCK_NORMS.json',{'XX':float(np.linalg.norm(Hsym[:6,:6])),'XU':float(np.linalg.norm(Hsym[:6,6:])),'UX':float(np.linalg.norm(Hsym[6:,:6])),'UU':float(np.linalg.norm(Hsym[6:,6:])),'full':float(np.linalg.norm(Hsym))})
 (R/'T1_LAYER_ARCHITECTURE_AUDIT.md').write_text('''# Certified architecture and indexing
Verified checkpoint has 16 transformer layers, hidden size 2048, intermediate size 1024, 64 experts and native top-8 routing. `model.model.layers[15].mlp` is the final zero-based block. Archived `model_io.py` directly indexes `model.model.layers[idx].mlp`; archived LAYERS=(7,15), so there is no indexing conflict.
Router: `mlp.gate.weight`, shape (64,2048), no bias. Expert output: `mlp.experts[e].down_proj.weight`, shape (2048,1024), no bias. Native softmax/top-8 routing is retained, with norm_topk_prob=false. All parameters are fixed; six output radial gains and six native gate-row radial factors implement the frozen subspace.
Prefix activations are cached because controls act only in the final-layer MLP; the prefix is mathematically independent of all 12 controls. Unchanged residual, final RMSNorm and LM head form the exact suffix. Direct radial scaling versus gain evaluation, zero logits and gradient precision are certified on Bank A in T1_ZERO_CONTROL_EQUIVALENCE.json and audit/T1_PRECISION_CHECK.json. CPU float64 comparison retains the native RMSNorm's float32 variance operation.
''')
 (R/'T1_OBJECTIVE_AUDIT.md').write_text('''# Certified native configured objective
Verified config: router_aux_loss_coef=0.001. Local OlmoeForCausalLM adds that coefficient times its native load_balancing_loss_func when router logits are requested with labels. Router-logit collection is explicitly enabled to retain this configured native auxiliary training term; the config's output_router_logits=false is an output/default execution flag, not a zero coefficient.
The objective is the native full-bank batch loss: mean causal cross entropy over 95 predicted tokens for each 96-token sequence, plus 0.001 times native auxiliary load balancing over all 16 layers and all 96 routing tokens per document. No tokenization changes: S96 is the first 96 archived token IDs of each 256-ID record.
Let N=16*96*bank_documents, f_e=(total top-8 assignments to expert e across all layers)/N and p_e=(sum full-softmax probabilities for expert e across all layers and tokens)/N. Auxiliary L_aux=64*sum_e f_e*p_e, exactly the local implementation's sum over top-k assignment ranks. Top-k memberships/counts are nondifferentiable as in native autograd; probabilities remain differentiable. Counts are recomputed at every perturbed state.
To avoid a batch-partition change in this nonadditive auxiliary, the prefix-layer probability sums and assignment counts are pooled over the complete bank. LM gradients/Hessians are token-weighted across chunks and the one whole-bank auxiliary gradient/Hessian is then added. This is not an average of independently normalized chunk auxiliary losses.
The configured coefficient and formula are recovered unambiguously from verified config and local model code, preserved under technical_reference/. A Bank-A sample agrees with the untouched native configured total loss; see equivalence audit. No auxiliary coefficient or objective was selected using Bank B.
Domainwise descriptive diagnostics use the same formula on each 32-document domain, compared to the single frozen H_A; they are not adjudication inputs.
''')
 log('Hessian certified: '+json.dumps(report));seal()

def cos(a,b):return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def summary(rows,amp):
 rr=[r for r in rows if r['amplitude']==amp];den=sum(r['response_norm']**2 for r in rr);ef=np.sqrt(sum(r['full_error']**2 for r in rr)/den);ed=np.sqrt(sum(r['no_mixed_error']**2 for r in rr)/den)
 pairs=[]
 for j in range(12):
  p=[r for r in rr if r['direction_id']==j];pairs.append({'direction_id':j,'full_mean_squared_error':np.mean([r['full_error']**2 for r in p]),'no_mixed_mean_squared_error':np.mean([r['no_mixed_error']**2 for r in p])})
 return {'amplitude':amp,'E_full':float(ef),'E_diag':float(ed),'R':float(ef/ed),'median_C_full':float(np.median([r['full_cosine'] for r in rr])),'direction_pairs_full_better':int(sum(p['full_mean_squared_error']<p['no_mixed_mean_squared_error'] for p in pairs)),'pairs':pairs,'median_support_turnover':float(np.median([r['support_turnover_fraction'] for r in rr])),'p95_support_turnover':float(np.quantile([r['support_turnover_fraction'] for r in rr],.95))}

def evaluate(bank):
 if bank=='B':
  sealdata=json.loads((R/'T0_PROTOCOL_SEAL.json').read_text());assert sealdata['production_freeze_certified'];assert all(sha(R/n)==h for n,h in sealdata['files'].items())
 chunks=load_chunks(bank);head,norm=load_suffix();H=np.load(R/'T2_HESSIAN_A_SYM.npy');diag=H.copy();diag[:6,6:]=0;diag[6:,:6]=0
 log(f'Measuring {bank} baseline gradient');loss0,g0,_,d0=measure(chunks,np.zeros(12),head,norm,domainwise=True);rows=[];responses=[];states=[];gs=[];drs={d:[] for d in d0};turn=[]
 np.save(R/f'audit/g_{bank}0.npy',g0);put(f'audit/{bank}_baseline_loss.json',{'loss':loss0})
 for a in [.01,.02,.04]:
  for j,d in enumerate(D):
   for s in [-1,1]:
    z=s*a*d;loss,g,_,dgs=measure(chunks,z,head,norm,domainwise=True);r=g-g0;pred=H@z;pd=diag@z;ef=np.linalg.norm(r-pred);ed=np.linalg.norm(r-pd);rn=np.linalg.norm(r);rt=routing(chunks,z)
    row={'amplitude':a,'direction_id':j,'sign':s,'response_norm':float(rn),'full_error':float(ef),'no_mixed_error':float(ed),'normalized_full_error':float(ef/max(rn,1e-30)),'normalized_no_mixed_error':float(ed/max(rn,1e-30)),'full_cosine':cos(r,pred),'no_mixed_cosine':cos(r,pd),**{k:v for k,v in rt.items() if k!='margin_conditioned'}}
    rows.append(row);responses.append(r);states.append(z);gs.append(g)
    for domain in drs:drs[domain].append(dgs[domain]-d0[domain])
    turn.append({'amplitude':a,'direction_id':j,'sign':s,**rt})
   log(f'{bank} amplitude {a} direction {j+1}/12 complete')
 np.savez_compressed(R/f'audit/{bank}_raw_gradients_responses.npz',g0=g0,gradients=gs,responses=responses,states=states,**{f'domain_response_{d}':v for d,v in drs.items()})
 sums=[summary(rows,a) for a in [.01,.02,.04]]
 if bank=='A':
  put('T4_BANK_A_SAMEBANK_SUMMARY.json',{'primary':False,'summaries':sums,'states':rows});return
 with (R/'T3_BANK_B_STATE_METRICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (R/'T3_SUPPORT_TURNOVER.csv').open('w') as f:
  out=[]
  for t in turn:
   for bin in t['margin_conditioned']:out.append({**{k:v for k,v in t.items() if k!='margin_conditioned'},'margin_bin_lower':bin['lo'],'margin_bin_upper':bin['hi'],'margin_bin_tokens':bin['tokens'],'margin_bin_turnover':bin['turnover']})
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 base=torch.cat([c['base_logits'] for c in chunks]).numpy();np.save(R/'audit/B_baseline_router_logits.npy',base)
 with (R/'T1_ROUTING_BASELINE_DIAGNOSTICS.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['bank','record_index','token_index','top8_experts','top8_probabilities','logit_margin_8_9'])
  ts=[c['base_logits'].to(DEVICE) for c in chunks];ps=[t.softmax(-1) for t in ts];ids=torch.cat([p.topk(8,-1).indices.cpu() for p in ps]);vals=torch.cat([t.topk(9,-1).values.cpu().double() for t in ts]);probs=torch.cat([p.cpu() for p in ps]);offset=0
  for c in chunks:
   for idx in c['indices']:
    for tok in range(96):
     ii=offset+tok;w.writerow(['B',idx,tok,json.dumps(ids[ii,:8].tolist()),json.dumps(probs[ii,ids[ii,:8]].tolist()),float(vals[ii,7]-vals[ii,8])])
    offset+=96
 primary=sums[1];rng=np.random.Generator(np.random.PCG64(2026103003));boot=[];pr=[r for r in rows if r['amplitude']==.02]
 for k in range(10000):
  inds=rng.integers(0,12,12);chosen=[pr[2*i+s] for i in inds for s in range(2)];rat=np.sqrt(sum(r['full_error']**2 for r in chosen)/sum(r['no_mixed_error']**2 for r in chosen));med=np.median([r['full_cosine'] for r in chosen]);boot.append([k,float(rat),float(med)])
 with (R/'T3_DIRECTION_BOOTSTRAP.csv').open('w') as f:w=csv.writer(f);w.writerow(['replicate','R','median_C_full']);w.writerows(boot)
 b=np.array(boot);primary['R_bootstrap_ci95']=np.quantile(b[:,1],[.025,.975]).tolist();primary['median_C_full_bootstrap_ci95']=np.quantile(b[:,2],[.025,.975]).tolist()
 asym=[]
 for j in range(12):
  rm=responses[24+2*j];rp=responses[24+2*j+1];asym.append({'direction_id':j,'sign_pair_anticorrelation_cosine':cos(rp,-rm),'even_to_odd_norm_ratio':float(np.linalg.norm(rp+rm)/max(np.linalg.norm(rp-rm),1e-30))})
 primary['sign_pair_diagnostics']=asym
 put('T3_BANK_B_PRIMARY_SUMMARY.json',{'primary':primary,'all_amplitudes':sums,'primary_operator':'H_A','scientific_evaluation_occurred':True})
 domainrows=[]
 for domain,rs in drs.items():
  for a in [.01,.02,.04]:
   ids=[k for k,row in enumerate(rows) if row['amplitude']==a];den=sum(np.linalg.norm(rs[k])**2 for k in ids);ef=np.sqrt(sum(np.linalg.norm(rs[k]-H@states[k])**2 for k in ids)/den);ed=np.sqrt(sum(np.linalg.norm(rs[k]-diag@states[k])**2 for k in ids)/den)
   domainrows.append({'domain':domain,'amplitude':a,'E_full':float(ef),'E_diag':float(ed),'R':float(ef/ed),'median_C_full':float(np.median([cos(rs[k],H@states[k]) for k in ids]))})
 with (R/'T3_DOMAINWISE_SUMMARY.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(domainrows[0]));w.writeheader();w.writerows(domainrows)
 files=[p for p in R.glob('T3_*') if p.is_file()]+[R/'audit/B_raw_gradients_responses.npz',R/'audit/B_baseline_router_logits.npy',R/'T1_ROUTING_BASELINE_DIAGNOSTICS.csv']
 put('audit/PRIMARY_RESULTS_SEAL.json',{'run_revision':4,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{str(p.relative_to(R)):sha(p) for p in files}})
 log('PRIMARY METRICS SEALED '+json.dumps(primary))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['certify','collect_B','primary','secondary']);args=p.parse_args()
 if args.stage=='certify':preprimary()
 elif args.stage=='collect_B':collect('B')
 elif args.stage=='primary':evaluate('B')
 elif args.stage=='secondary':
  assert (R/'audit/PRIMARY_RESULTS_SEAL.json').is_file();evaluate('A')