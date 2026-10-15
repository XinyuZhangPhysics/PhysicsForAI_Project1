"""Two-amplitude pre-Top-k action and paired stratified endpoint bootstrap."""
import hashlib
import numpy as np
def logits_action(logits,bias,amplitude):
    logits=np.asarray(logits,dtype=np.float32);bias=np.asarray(bias,dtype=np.float32)
    if logits.shape[-1]!=64 or bias.shape!=(64,):raise ValueError('64 expert logits required')
    return logits+np.float32(amplitude)*bias
def endpoint(losses,draws=10000):
    a=np.asarray(losses,dtype=np.float64)
    if a.shape!=(41,7,64) or not np.isfinite(a).all():raise ValueError('expected 41 aligned arms, 7 domains, 64 documents')
    utilities=a[0].mean()-a.mean((1,2));delta=a[2:].mean(0)-a[1];gain=a[0]-a[1]
    seed=int.from_bytes(hashlib.sha256(b'PROJECT1-Q3-BOOTSTRAP-v173').digest()[:8],'little')
    rng=np.random.Generator(np.random.PCG64(seed));boot=np.empty((draws,2));ds=np.arange(7)[:,None]
    for i in range(draws):
        ids=rng.integers(0,64,(7,64));boot[i]=[delta[ds,ids].mean(),gain[ds,ids].mean()]
    ci=np.quantile(boot,[.025,.975],axis=0,method='linear')
    return {'D':float(utilities[1]-utilities[2:].mean()),'D_CI95':ci[:,0].tolist(),'candidate_utility':float(utilities[1]),'candidate_vs_baseline_CI95':ci[:,1].tolist(),'candidate_rank':1+int(np.count_nonzero(utilities[2:]>utilities[1]))}
