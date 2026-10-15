"""Signed child-sector router pushforward with frozen normalization."""
import numpy as np
def softmax(x):
    z=np.asarray(x)-np.max(x,axis=-1,keepdims=True);e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)
def child_pushforward(ax,ay,tx,ty,e,s):
    e=np.asarray(e);p,g=e.shape
    if not np.allclose(e.mean(axis=1),0):raise ValueError('child-row-centered direction required')
    weight=softmax(s*np.asarray(ax))*softmax(s*np.asarray(ay))
    row=(tx+ty)*(weight@e)/(p*g*g)
    return np.tile(row,p)
def mixed_slope(ax,ay,tx,ty,e):
    e=np.asarray(e);p,g=e.shape
    if not np.allclose(e.sum(axis=0),0) or not np.allclose(e.sum(axis=1),0):raise ValueError('mixed sector required')
    row=(tx+ty)*((np.asarray(ax)+ay)@e)/(p**3*g*g)
    return np.tile(row,p)
def paired_residual(measured,predicted):return np.asarray(measured)-predicted
