"""Six-control tangent and physical operator gates, fixed selected support.\n\nThe local Gram uses the mean squared response per output coordinate; for the paper tensors this is exactly division by 256*2048. The objective Gram is formed from signed document-level objective contractions and is not divided by that response-coordinate factor.\n"""
import numpy as np
def soft_tangent(probabilities,selected,expert_outputs,gain,bias):
    # Mean bias uses all 64 full-softmax probabilities; Top-8 is unnormalized.
    p=np.asarray(probabilities);idx=np.asarray(selected);out=np.asarray(expert_outputs)
    mean=p@np.asarray(bias)
    w=np.take_along_axis(p,idx,axis=-1)
    factor=w*(np.asarray(gain)[idx]+np.asarray(bias)[idx]-mean[:,None])
    return np.einsum('tk,tkh->th',factor,out)
def gram_operators(tangents,adjoint):
    """Return local-response and objective Gram matrices.

    ``tangents`` is control x response-coordinate. In the manuscript each
    response has shape (256, 2048), so ``u.shape[1] == 256*2048`` and the
    local matrix is normalized by that factor. The objective matrix is the
    outer product of signed objective contractions and receives no such
    normalization.
    """
    t=np.asarray(tangents);u=t.reshape(t.shape[0],-1)
    local=u@u.T/u.shape[1];signed=u@np.asarray(adjoint).ravel()
    return local,np.outer(signed,signed)
def physical_operator(b,h):
    e,v=np.linalg.eigh(h)
    if np.any(e<=0):raise ValueError('control metric must be positive definite')
    hi=(v/np.sqrt(e))@v.T;return hi@b@hi
def shape_cosine(a,b):return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def effective_dimension(a):return float(np.trace(a)**2/np.sum(a*a))
def split_gate(a,b):
    cos=shape_cosine(a,b);d0=effective_dimension(a);d1=effective_dimension(b);dr=abs(d0-d1)/((d0+d1)/2)
    return {'shape_cosine':cos,'relative_deff_difference':dr,'stable':cos>=.90 and dr<=.25}
def scalar_gate(baseline,terminal,c):
    prediction=c*baseline;error=float(np.linalg.norm(terminal-prediction)/np.linalg.norm(terminal));cos=shape_cosine(terminal,prediction)
    return {'scale_error':error,'shape_cosine':cos,'adequate':error<=.20 and cos>=.95}
