"""Equivalent AD for repeated expert gain selection without MPS atomic accumulation.

Forward is the original alpha[sel]. Backward uses the exact one-hot transpose
linear map. Its differentiable matmul backward supports the same second-order AD.
No model, loss, top-k, coordinates, states, or archived data are changed.
"""
import inspect
import torch
import torch.nn.functional as F
class DeterministicExpertSelect(torch.autograd.Function):
 @staticmethod
 def forward(ctx,alpha,sel):
  ctx.save_for_backward(sel);ctx.width=alpha.numel()
  return alpha[sel]
 @staticmethod
 def backward(ctx,grad_output):
  (sel,)=ctx.saved_tensors
  onehot=F.one_hot(sel.reshape(-1),ctx.width).to(grad_output.dtype)
  return onehot.T@grad_output.reshape(-1),None

def install(engine):
 source=inspect.getsource(engine.suffix)
 old='alpha[sel,None]'
 assert source.count(old)==1
 source=source.replace(old,'deterministic_expert_select(alpha,sel)[:,:,None]')
 engine.deterministic_expert_select=DeterministicExpertSelect.apply
 exec(compile(source,__file__,'exec'),engine.__dict__)
 return {'original_expression':old,'replacement_expression':'deterministic_expert_select(alpha,sel)[:,:,None]','forward':'Identical indexed values','backward':'One-hot transpose matmul instead of nondeterministic MPS index_put atomic accumulation','differentiation':'second-order autograd retained'}