"""Common-origin batch maps and finite-horizon closures; no model capture."""
import numpy as np
from scipy.linalg import logm,expm
from .formation import CrossOperator
def local_map(c,mu,lambda_,eta_m,eta_u):
    c=np.asarray(c);du,dm=c.shape
    return np.block([[(1-eta_m*mu)*np.eye(dm),eta_m*c.T],[eta_u*c,(1-eta_u*lambda_)*np.eye(du)]])
def log_cumulants(maps):
    ks=[logm(x) for x in maps]
    if any(np.max(np.abs(np.imag(k)))>1e-10 for k in ks):raise ValueError('real map logarithm required')
    ks=[k.real for k in ks];o1=sum(ks);o2=np.zeros_like(o1)
    for m in range(len(ks)):
        for n in range(m):o2+=(ks[m]@ks[n]-ks[n]@ks[m])/2
    return o1,o2
def closures(maps,reference_maps):
    maps=np.asarray(maps);n=len(maps);ordered=np.eye(maps.shape[-1])
    for x in maps:ordered=x@ordered
    o1,o2=log_cumulants(maps)
    return {'Phi_refmean':np.linalg.matrix_power(np.mean(reference_maps,axis=0),n),'Phi_seqmean':np.linalg.matrix_power(maps.mean(0),n),'Phi_log':expm(o1),'Phi_log+comm':expm(o1+o2),'Phi_ord':ordered}
def closure_error(predicted,observed,initial):
    den=np.linalg.norm(np.asarray(observed)-initial)
    if den==0:raise ValueError('zero displacement denominator')
    return float(np.linalg.norm(np.asarray(predicted)-observed)/den)
def reconstruction(residual,correction):
    r=np.asarray(residual);c=np.asarray(correction);den=np.sum(r*r)
    return {'cosine':float(np.sum(r*c)/(np.linalg.norm(r)*np.linalg.norm(c))),'projection_coefficient':float(np.sum(r*c)/np.sum(c*c)),'relative_remainder':float(np.linalg.norm(r-c)/np.linalg.norm(r)),'explained_squared_norm':float(1-np.sum((r-c)**2)/den)}
