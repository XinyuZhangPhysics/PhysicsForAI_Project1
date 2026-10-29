from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, socket, sys, time, traceback
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1')
def _deny(*a,**kw): raise RuntimeError('NETWORK_PROHIBITED_BY_FROZEN_PROTOCOL')
socket.create_connection=_deny
try:
    socket.socket.connect=_deny; socket.socket.connect_ex=_deny
except Exception: pass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
from generic_core import sha256_file, array_sha256, make_generic_direction, central_norm_np, raw_norm_np, aligned_central_amplitude

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'protocol.json').read_text())
MU=float(P['mu']); ETA_X=float(P['eta_X']); ETA_U=ETA_X; H=int(P['horizon'])
F0_PATH=ROOT/'inputs/F0.npy'; SPEC_PATH=ROOT/'inputs/P2H_CROSS_HESSIAN_SPECTRUM.npz'
RUN_STATE={'stage':'IMPORT_COMPLETE','gradient_evaluations':0,'failures':[],'rows':[],'output':None,'model_dir':None,'started_utc':None}

def jwrite(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def norm_t(x): return float(torch.linalg.vector_norm(x.double()))
def dot_t(a,b): return float((a.double()*b.double()).sum())
def combined_dot(X1,U1,X2,U2): return dot_t(X1,X2)+dot_t(U1,U2)
def combined_norm(X,U): return math.hypot(norm_t(X),norm_t(U))
def central_norm_t(X,U): return math.sqrt(MU)*combined_norm(X,U)
def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class PrivateMoE(nn.Module):
    def __init__(self,dense):
        super().__init__(); self.dense=dense; dense.requires_grad_(False)
        self.M=nn.Parameter(torch.zeros_like(dense.down_proj.weight)); self.U=nn.Parameter(torch.zeros(576))
        self.register_buffer('c',torch.tensor([.5,.5,-.5,-.5]))
    def forward(self,h):
        phi=F.silu(self.dense.gate_proj(h))*self.dense.up_proj(h); y=self.dense.down_proj(phi)
        p=torch.softmax((h@self.U).unsqueeze(-1)*self.c,dim=-1)
        return y+(p*self.c).sum(-1,keepdim=True)*F.linear(phi,self.M)

def context(model,h):
    pos=torch.arange(h.shape[1])
    if h.ndim==3: pos=pos.unsqueeze(0).expand(h.shape[0],-1)
    mask=create_causal_mask(config=model.config,inputs_embeds=h,attention_mask=None,past_key_values=None,position_ids=pos)
    rot=model.model.rotary_emb(h,position_ids=pos)
    return dict(attention_mask=mask,position_embeddings=rot,position_ids=pos,use_cache=False)

def prefix_one(model,row):
    with torch.no_grad():
        h=model.model.embed_tokens(row[None]); kw=context(model,h)
        for layer in model.model.layers[:20]: h=layer(h,**kw)
    return h.detach()

def suffix(model,h,ids):
    for layer in model.model.layers[20:]: h=layer(h,**context(model,h))
    logits=model.lm_head(model.model.norm(h)).float()
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))

def capture_operator(model,dense,ids):
    hs=[]; ps=[]; gs=[]
    for n,row in enumerate(ids):
        cache={}
        def hook(mod,args,y):
            h=args[0].detach(); cache['h']=h; cache['phi']=(F.silu(mod.gate_proj(h))*mod.up_proj(h)).detach()
            cache['y']=y.detach().requires_grad_(True); return cache['y']
        handle=dense.register_forward_hook(hook)
        try:
            loss=model(input_ids=row[None],labels=row[None],use_cache=False).loss
            g=torch.autograd.grad(loss,cache['y'])[0]/len(ids)
        finally: handle.remove()
        hs.append(cache['h'].reshape(-1,576)); ps.append(cache['phi'].reshape(-1,1536)); gs.append(g.detach().reshape(-1,576))
        if (n+1)%8==0: print(f'operator capture {n+1}/{len(ids)}',flush=True)
    return torch.cat(hs),torch.cat(ps),torch.cat(gs)

class FrozenCOperator:
    # Archived C = -H_UX. This sign convention is verified against the stored singular pair.
    def __init__(self,h,p,g,coef=.25): self.h=h; self.p=p; self.g=g; self.coef=coef
    def ct(self,u): return -self.coef*(self.g.T@((self.h@u).unsqueeze(-1)*self.p))
    def c(self,m): return -self.coef*(self.h.T@(self.g*F.linear(self.p,m)).sum(-1))

def discover_model():
    commit=P['model_commit']; candidates=[]
    if os.environ.get('PFAI_SMOLLM2_DIR'): candidates.append(Path(os.environ['PFAI_SMOLLM2_DIR']).expanduser())
    candidates.append(Path(P['preferred_model_path']))
    candidates += [
        Path.home()/'.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots'/commit,
        Path.home()/'.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M'/commit,
    ]
    seen=set()
    for p in candidates:
        p=p.resolve() if p.exists() else p
        if str(p) in seen: continue
        seen.add(str(p))
        if (p/'config.json').exists() and any((p/x).exists() for x in ['model.safetensors','model.safetensors.index.json','pytorch_model.bin']): return p
    roots=[Path.home()/'AI_Workspace/ResearchModels',Path.home()/'.cache/huggingface/hub']
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob(commit):
            if p.is_dir() and (p/'config.json').exists(): return p
    raise FileNotFoundError('Exact SmolLM2-135M revision not found. Set PFAI_SMOLLM2_DIR to the local revision directory; no network fallback is permitted.')

def assign(moe,state):
    with torch.no_grad(): moe.M.copy_(state[0]); moe.U.copy_(state[1])

def bank_gradient(model,moe,ids,hidden,state,lam):
    assign(moe,state); gm=torch.zeros_like(moe.M,dtype=torch.float64); gu=torch.zeros_like(moe.U,dtype=torch.float64); losses=[]
    for row,h in zip(ids,hidden):
        loss=suffix(model,h,row[None]); a,b=torch.autograd.grad(loss,(moe.M,moe.U)); gm+=a.detach().double(); gu+=b.detach().double(); losses.append(float(loss.detach()))
    gm/=len(ids); gu/=len(ids); gm+=MU*state[0].double(); gu+=lam*state[1].double()
    return gm,gu,float(np.mean(losses))

def linear_step(state,op,lam):
    X,U=state
    # H=-C, hence -eta H = +eta C in the update map.
    return ((1-ETA_X*MU)*X + ETA_X*op.ct(U), (1-ETA_U*lam)*U + ETA_U*op.c(X))

def cubic_diag(U,h,delta):
    a=h@U
    exact=.5*torch.tanh(a/2); linear=a/4; rem=exact-linear
    abs_a=a.abs(); denom=(abs_a.pow(3)/48).clamp_min(delta)
    ratio=rem.abs()/denom
    rel=rem.abs()/(linear.abs()+delta)
    return dict(max_abs_router_field=float(abs_a.max()),rms_router_field=float(torch.sqrt(torch.mean(a.double().square()))),max_abs_tanh_remainder=float(rem.abs().max()),max_remainder_over_cubic_bound=float(ratio.max()),rms_relative_tanh_remainder=float(torch.sqrt(torch.mean(rel.double().square()))))

def project_metrics(state,pred,z0,v,u,delta):
    X,U=state; XL,UL=pred; X0,U0=z0
    dX=X-X0; dU=U-U0; dXL=XL-X0; dUL=UL-U0
    obs_inc=combined_norm(dX,dU); lin_inc=combined_norm(dXL,dUL)
    abs_err=combined_norm(dX-dXL,dU-dUL); err=abs_err/(lin_inc+delta)
    cos=None; reasons=[]
    if obs_inc<delta: reasons.append('OBS_INCREMENT_BELOW_DELTA')
    if lin_inc<delta: reasons.append('LINEAR_INCREMENT_BELOW_DELTA')
    if not reasons: cos=combined_dot(dX,dU,dXL,dUL)/(obs_inc*lin_inc)
    rec=dict(error=err,prediction_error_abs_raw=abs_err,prediction_error_abs_central=math.sqrt(MU)*abs_err,cosine=cos,cosine_valid=(not reasons),cosine_reason='OK' if not reasons else ';'.join(reasons),state_raw=combined_norm(X,U),linear_state_raw=combined_norm(XL,UL),increment_raw=obs_inc,linear_increment_raw=lin_inc,state_central=central_norm_t(X,U),linear_state_central=central_norm_t(XL,UL),increment_central=central_norm_t(dX,dU),linear_increment_central=central_norm_t(dXL,dUL),linear_X_sha256=tensor_sha(XL),linear_U_sha256=tensor_sha(UL))
    mx=[]; ux=[]; mlx=[]; ulx=[]
    for k in range(v.shape[-1]):
        mk=dot_t(X,v[:,:,k]); uk=dot_t(U,u[:,k]); ml=dot_t(XL,v[:,:,k]); ul=dot_t(UL,u[:,k])
        ce=math.hypot(mk-ml,uk-ul); cr=ce/(math.hypot(ml,ul)+delta)
        rec.update({f'm{k+1}_obs':mk,f'u{k+1}_obs':uk,f'm{k+1}_lin':ml,f'u{k+1}_lin':ul,f'pair{k+1}_coeff_error_abs':ce,f'pair{k+1}_coeff_error_rel':cr})
        mx.append(mk); ux.append(uk); mlx.append(ml); ulx.append(ul)
    Xr=X.clone().double(); Ur=U.clone().double(); XLr=XL.clone().double(); ULr=UL.clone().double()
    for k in range(v.shape[-1]):
        Xr-=mx[k]*v[:,:,k].double(); Ur-=ux[k]*u[:,k].double(); XLr-=mlx[k]*v[:,:,k].double(); ULr-=ulx[k]*u[:,k].double()
    rec['residual_top4_raw']=math.hypot(norm_t(Xr),norm_t(Ur)); rec['linear_residual_top4_raw']=math.hypot(norm_t(XLr),norm_t(ULr))
    rec['residual_top4_central']=math.sqrt(MU)*rec['residual_top4_raw']; rec['linear_residual_top4_central']=math.sqrt(MU)*rec['linear_residual_top4_raw']
    return rec

def finite_or_none(x):
    if isinstance(x,float) and (math.isnan(x) or math.isinf(x)): return None
    return x

def write_csv(path,rows):
    keys=list(rows[0].keys())
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def summarize(rows):
    end=[r for r in rows if int(r['step'])==H]
    out={}
    for side in ['stable','unstable']:
        out[side]={}
        for amp in ['small','large']:
            ss=[r for r in end if r['side']==side and r['amplitude']==amp]
            def stats(key):
                a=np.array([float(r[key]) for r in ss if r[key] not in ('',None) and math.isfinite(float(r[key]))])
                return dict(n=int(a.size),median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)),max=float(np.max(a)),min=float(np.min(a))) if a.size else dict(n=0)
            out[side][amp]={'endpoint_error':stats('error'),'endpoint_cosine':stats('cosine'),'endpoint_state_central':stats('state_central'),'endpoint_linear_state_central':stats('linear_state_central')}
    return out

def plot_pdf(path,endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for metric,title,ylabel in [('error','Endpoint linearization error','e_28'),('cosine','Endpoint increment cosine','cosine')]:
            fig,ax=plt.subplots()
            x=[];y=[];labels=[]
            idx=0
            for side in ['stable','unstable']:
                for amp in ['small','large']:
                    vals=[r for r in endpoints if r['side']==side and r['amplitude']==amp]
                    for r in vals:
                        jitter=(int(r['direction'])-4.5)*0.07
                        x.append(idx+jitter); y.append(float(r[metric]) if r[metric] not in ('',None) else np.nan); labels.append(f'{side[0].upper()}-{amp[0].upper()}-{r["direction"]}')
                    idx+=1
            ax.scatter(x,y); ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel('group index with deterministic direction jitter: stable-small, stable-large, unstable-small, unstable-large'); ax.grid(True,alpha=.25); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots()
        for side in ['stable','unstable']:
            for d in range(1,9):
                a={r['amplitude']:float(r['error']) for r in endpoints if r['side']==side and int(r['direction'])==d}
                ax.scatter(a['small'],a['large'])
        lo=min([float(r['error']) for r in endpoints]); hi=max([float(r['error']) for r in endpoints]); ax.plot([lo,hi],[lo,hi]); ax.set_xlabel('small-amplitude e_28'); ax.set_ylabel('large-amplitude e_28'); ax.set_title('Amplitude comparison (all directions, both sides)'); ax.grid(True,alpha=.25); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)

def main():
    RUN_STATE['stage']='ARGUMENT_PARSE'
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--smoke-only',action='store_true'); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True); RUN_STATE['output']=str(out); RUN_STATE['started_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    forbidden=['generic_local_dynamics_long.csv','generic_local_dynamics_summary.json','generic_local_dynamics_operator.npz']
    if not args.smoke_only and any((out/x).exists() for x in forbidden): raise FileExistsError('Refusing to overwrite an existing scientific result directory')
    t0=time.monotonic(); RUN_STATE['stage']='RUNTIME_SETUP'; torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
    model_dir=discover_model(); RUN_STATE['model_dir']=str(model_dir); RUN_STATE['stage']='MODEL_LOAD'; print(f'MODEL {model_dir}',flush=True)
    ids=torch.from_numpy(np.load(F0_PATH,allow_pickle=False)); spec=np.load(SPEC_PATH,allow_pickle=False); s=torch.from_numpy(spec['s'].copy()); u=torch.from_numpy(spec['u'].copy()); v=torch.from_numpy(spec['v'].copy())
    model=AutoModelForCausalLM.from_pretrained(str(model_dir),dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval(); model.requires_grad_(False)
    if (model.config.hidden_size,model.config.intermediate_size,model.config.num_hidden_layers)!=(576,1536,30): raise RuntimeError('model architecture mismatch')
    dense=model.model.layers[20].mlp
    RUN_STATE['stage']='OPERATOR_CAPTURE'
    h,phi,g=capture_operator(model,dense,ids); op=FrozenCOperator(h,phi,g)
    rv=norm_t(op.c(v[:,:,0])-s[0]*u[:,0])/float(s[0]); ru=norm_t(op.ct(u[:,0])-s[0]*v[:,:,0])/float(s[0])
    if max(rv,ru)>5e-5: raise RuntimeError(f'archived operator smoke mismatch: c(v)={rv}, ct(u)={ru}')
    # Verify the mode-aligned 2D discrete map, including sign convention, on both sides.
    aligned_map_smoke=[]
    for side,eps in [('stable',-0.08),('unstable',0.08)]:
        lam=MU*(1-eps); rho=float(s[0])
        A2=np.array([[-ETA_X*MU, ETA_X*rho],[ETA_U*rho,-ETA_U*lam]],dtype=np.float64)
        vals,vecs=np.linalg.eig(A2); k=int(np.argmax(vals.real)); sigma=float(vals[k].real); e=vecs[:,k].real; e=e/np.linalg.norm(e)
        z=(float(P['a0_aligned_raw'])*e[0]*v[:,:,0],float(P['a0_aligned_raw'])*e[1]*u[:,0])
        z1=linear_step(z,op,lam)
        expected=((1+sigma)*z[0],(1+sigma)*z[1])
        er=combined_norm(z1[0]-expected[0],z1[1]-expected[1])/(combined_norm(expected[0],expected[1])+1e-30)
        if er>5e-5: raise RuntimeError(f'aligned map smoke mismatch {side}: {er}')
        aligned_map_smoke.append(dict(side=side,epsilon=eps,sigma_plus=sigma,gamma_disc=math.log(abs(1+sigma)),e_M=float(e[0]),e_U=float(e[1]),relative_map_error=er))
    # Verify all archived top-four singular pairs to diagnose exact model/data binding.
    pair_res=[]
    for k in range(4):
        pair_res.append(dict(k=k+1,c_v=norm_t(op.c(v[:,:,k])-s[k]*u[:,k])/float(s[k]),ct_u=norm_t(op.ct(u[:,k])-s[k]*v[:,:,k])/float(s[k])))
    RUN_STATE['stage']='OPERATOR_SMOKE_PASS'
    # Save exact matrix-free factorization before trajectories.
    Aalign=aligned_central_amplitude(float(P['a0_aligned_raw']),MU); amps={'large':Aalign,'small':Aalign/2}; delta=1e-12*Aalign
    directions=[]
    for seed in P['direction_seeds']:
        Xd,Ud=make_generic_direction(int(seed),(576,1536),(576,),MU)
        directions.append((Xd.astype(np.float32),Ud.astype(np.float32)))
    np.savez_compressed(out/'generic_local_dynamics_operator.npz',h=h.numpy(),phi=phi.numpy(),g=g.numpy(),s=s.numpy(),u=u.numpy(),v=v.numpy(),direction_seeds=np.asarray(P['direction_seeds'],dtype=np.int64),X_directions=np.stack([x for x,_ in directions]),U_directions=np.stack([y for _,y in directions]),mu=np.asarray(MU),eta_X=np.asarray(ETA_X),eta_U=np.asarray(ETA_U),A_large=np.asarray(Aalign),A_small=np.asarray(Aalign/2))
    smoke=dict(status='PASS',model_dir=str(model_dir),model_commit=P['model_commit'],F0_file_sha256=sha256_file(F0_PATH),F0_array_sha256=array_sha256(ids.numpy()),spectrum_file_sha256=sha256_file(SPEC_PATH),pair_residuals=pair_res,aligned_map_smoke=aligned_map_smoke,operator_factorization_sha256=sha256_file(out/'generic_local_dynamics_operator.npz'),eta_X_mu=ETA_X*MU,A_align=Aalign,delta=delta)
    jwrite(out/'SMOKE_TEST.json',smoke); print('SMOKE_TEST PASS',json.dumps(smoke,indent=2),flush=True)
    if args.smoke_only: return
    RUN_STATE['stage']='GENERIC_TRAJECTORIES'
    moe=PrivateMoE(dense); model.model.layers[20].mlp=moe
    if [n for n,p in model.named_parameters() if p.requires_grad]!=['model.layers.20.mlp.M','model.layers.20.mlp.U']: raise RuntimeError('trainable parameter mismatch')
    hidden=[prefix_one(model,row) for row in ids]
    rows=[]; failures=[]; gradient_evals=0
    for side,eps in [('stable',-0.08),('unstable',0.08)]:
        lam=MU*(1-eps)
        for di,(Xd,Ud) in enumerate(directions,1):
            for aname,A in [('small',amps['small']),('large',amps['large'])]:
                X0=torch.from_numpy((A*Xd).astype(np.float32)); U0=torch.from_numpy((A*Ud).astype(np.float32)); z0=(X0,U0); state=(X0.clone(),U0.clone()); pred=(X0.clone(),U0.clone())
                # Diagnostic-only coupling activity. These quantities never gate, filter, reorder, or exclude a direction.
                CX0=op.c(X0); CTU0=op.ct(U0)
                diag0=combined_norm(-ETA_X*MU*X0,-ETA_U*lam*U0)
                offdiag0=combined_norm(ETA_X*CTU0,ETA_U*CX0)
                coupling_diag=dict(coupling_CX0_raw=norm_t(CX0),coupling_CT_U0_raw=norm_t(CTU0),first_step_diag_raw=diag0,first_step_offdiag_raw=offdiag0,coupling_activity_ratio=offdiag0/(diag0+delta),coupling_diagnostic_only=True)
                # Both sides intentionally share these exact initial bytes for each direction/amplitude.
                for n in range(H+1):
                    rec=dict(side=side,epsilon=eps,lambda_=lam,direction=di,seed=int(P['direction_seeds'][di-1]),amplitude=aname,A=float(A),step=n); rec.update(coupling_diag)
                    met=project_metrics(state,pred,z0,v,u,delta); rec.update({k:finite_or_none(val) for k,val in met.items()})
                    cd=cubic_diag(state[1],h,delta); rec.update(cd); rec['F0_NLL_before_update']=''
                    if n<H:
                        try:
                            gm,gu,nll=bank_gradient(model,moe,ids,hidden,state,lam); gradient_evals+=1; RUN_STATE['gradient_evaluations']=gradient_evals; rec['F0_NLL_before_update']=nll
                            if not torch.isfinite(gm).all() or not torch.isfinite(gu).all(): raise FloatingPointError('nonfinite gradient')
                            state=((state[0].double()-ETA_X*gm).float(),(state[1].double()-ETA_U*gu).float())
                            pred=linear_step(pred,op,lam)
                        except Exception as e:
                            failures.append(dict(side=side,direction=di,amplitude=aname,step=n,error_type=type(e).__name__,error=str(e))); RUN_STATE['failures']=failures; rows.append(rec); RUN_STATE['rows']=rows; raise
                    rows.append(rec); RUN_STATE['rows']=rows
                print(f'COMPLETE {side} direction={di} amp={aname} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    RUN_STATE['stage']='WRITE_RESULTS'
    write_csv(out/'generic_local_dynamics_long.csv',rows)
    endpoints=[r for r in rows if int(r['step'])==H]; write_csv(out/'generic_local_dynamics_endpoints.csv',endpoints)
    jwrite(out/'generic_local_dynamics_summary.json',summarize(rows))
    plot_pdf(out/'Fig_generic_local_dynamics_diagnostic.pdf',endpoints)
    outputs=['generic_local_dynamics_long.csv','generic_local_dynamics_endpoints.csv','generic_local_dynamics_operator.npz','generic_local_dynamics_summary.json','Fig_generic_local_dynamics_diagnostic.pdf','SMOKE_TEST.json']
    manifest=dict(status='COMPLETE',execution_stage='COMPLETE',scientific_protocol_unchanged=True,diagnostic_only_fields=['coupling_CX0_raw','coupling_CT_U0_raw','first_step_diag_raw','first_step_offdiag_raw','coupling_activity_ratio'],protocol_id=P['protocol_id'],model_dir=str(model_dir),model_commit=P['model_commit'],layer_zero_based=20,F0_shape=list(ids.shape),mu=MU,eta_X=ETA_X,eta_U=ETA_U,eta_X_mu=ETA_X*MU,epsilon=P['epsilon'],horizon=H,seeds=P['direction_seeds'],direction_rng=P['direction_rng'],amplitudes=amps,delta=delta,gradient_evaluations=gradient_evals,expected_gradient_evaluations=2*8*2*H,pair_residuals=pair_res,failures=failures,elapsed_seconds=time.monotonic()-t0,runtime=dict(python=sys.executable,python_version=sys.version,platform=platform.platform(),torch=torch.__version__,transformers=__import__('transformers').__version__,numpy=np.__version__,threads=8,backend='CPU',deterministic=True),input_sha256={'F0.npy':sha256_file(F0_PATH),'P2H_CROSS_HESSIAN_SPECTRUM.npz':sha256_file(SPEC_PATH),'P2J_FROZEN_CELL_PREDICTIONS.csv':sha256_file(ROOT/'inputs/P2J_FROZEN_CELL_PREDICTIONS.csv'),'P2J_TRAINING_COMPLETION.json':sha256_file(ROOT/'inputs/P2J_TRAINING_COMPLETION.json'),'P2J_AMPLITUDE_GATE.csv':sha256_file(ROOT/'inputs/P2J_AMPLITUDE_GATE.csv')},output_sha256={x:sha256_file(out/x) for x in outputs})
    jwrite(out/'generic_local_dynamics_manifest.json',manifest)
    print('GENERIC_LOCAL_DYNAMICS COMPLETE',flush=True)

def write_stop_manifest(exc):
    out=Path(RUN_STATE['output']) if RUN_STATE.get('output') else ROOT/'results/generic_local_dynamics_2026-10-27'
    out.mkdir(parents=True,exist_ok=True)
    rows=RUN_STATE.get('rows') or []
    partial=None
    if rows:
        partial=out/'generic_local_dynamics_partial.csv'
        try: write_csv(partial,rows)
        except Exception: partial=None
    manifest={
        'status':'STOP',
        'protocol_id':P.get('protocol_id'),
        'scientific_protocol_unchanged':True,
        'execution_stage':RUN_STATE.get('stage'),
        'error_type':type(exc).__name__,
        'error':str(exc),
        'gradient_evaluations':int(RUN_STATE.get('gradient_evaluations',0)),
        'expected_gradient_evaluations':2*8*2*H,
        'failures':RUN_STATE.get('failures',[]),
        'model_dir':RUN_STATE.get('model_dir'),
        'started_utc':RUN_STATE.get('started_utc'),
        'stopped_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'partial_output_sha256': ({partial.name:sha256_file(partial)} if partial and partial.exists() else {}),
        'runtime':{'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__},
    }
    jwrite(out/'generic_local_dynamics_manifest.json',manifest)

if __name__=='__main__':
    try: main()
    except Exception as e:
        traceback.print_exc()
        try: write_stop_manifest(e)
        except Exception: traceback.print_exc()
        sys.exit(1)
