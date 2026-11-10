import torch
import torch.nn.functional as F
class FixedProbabilityGather(torch.autograd.Function):
 @staticmethod
 def forward(ctx,probs,sel):
  ctx.save_for_backward(sel);ctx.width=probs.shape[1];return probs.gather(1,sel)
 @staticmethod
 def backward(ctx,grad_output):
  (sel,)=ctx.saved_tensors;onehot=F.one_hot(sel,ctx.width).to(grad_output.dtype);return (onehot*grad_output[:,:,None]).sum(1),None

def install(engine):
 native_suffix=engine.suffix;native_aux=engine.aux
 def suffix_fixed(z,ch,head,norm):
  CC=engine.C if z.device.type=="mps" else engine.C64;alpha=1+CC@z[:6];beta=1+CC@z[6:];logits=ch["base_logits"]*beta
  probs=logits.softmax(-1);sel=ch["base_logits"].topk(8,-1).indices;weights=FixedProbabilityGather.apply(probs,sel)
  slot=torch.searchsorted(ch["union"].contiguous(),sel.contiguous());assert bool(torch.all(slot<ch["union"].shape[1])) and bool(torch.all(ch["union"].gather(1,slot)==sel))
  v=ch["v"].gather(1,slot[:,:,None].expand(-1,-1,2048));alpha_sel=engine.deterministic_expert_select(alpha,sel)
  y=(v*alpha_sel[:,:,None]*weights[:,:,None]).sum(1).reshape(ch["residual"].shape)+ch["residual"];y=norm(y);out=head(y)
  loss=F.cross_entropy(out[:,:-1,:].reshape(-1,out.shape[-1]),ch["ids"][:,1:].reshape(-1));return loss,logits,sel,out
 def aux_fixed(z,chunks,hessian=False):
  b=torch.cat([c["base_logits"] for c in chunks]).double();N=b.shape[0]*16;prefix_prob=sum(c["prefix_prob"] for c in chunks);prefix_count=sum(c["prefix_count"] for c in chunks)
  zz=torch.tensor(z,dtype=torch.float64,requires_grad=True);logits=b*(1+engine.C64@zz[6:]);p=logits.softmax(-1)
  with torch.no_grad():actual=torch.cat([c["base_logits"].to(device=engine.DEVICE,dtype=engine.DTYPE).topk(8,-1).indices.cpu() for c in chunks])
  count=prefix_count+F.one_hot(actual,64).sum((0,1)).double();loss=.001*64*((count/N)*((prefix_prob+p.sum(0))/N)).sum()
  g=torch.autograd.grad(loss,zz,create_graph=hessian)[0];H=torch.stack([torch.autograd.grad(g[j],zz,retain_graph=True)[0] for j in range(12)]) if hessian else None
  return float(loss.detach()),g.detach().numpy(),None if H is None else H.detach().numpy()
 engine.suffix_fixed_membership=suffix_fixed;engine.aux_fixed_membership=aux_fixed;return native_suffix,native_aux
