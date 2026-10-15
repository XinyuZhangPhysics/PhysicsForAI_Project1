"""Copied expert-index fields and the two-sided response rule."""
import numpy as np
def projectors(parents, children):
    jp=np.ones((parents,parents))/parents; jc=np.ones((children,children))/children
    pp=np.eye(parents)-jp; pc=np.eye(children)-jc
    return np.kron(pp,jc),np.kron(jp,pc),np.kron(pp,pc)
def center(field):
    field=np.asarray(field);return field-field.mean(axis=-1,keepdims=True)
def copied_fields(alpha_x, alpha_y, child_kernel):
    g=len(child_kernel);q=np.repeat(np.asarray(alpha_x)*alpha_y/g**2,g)
    k=np.tile(np.asarray(child_kernel),len(alpha_x))
    return center(q),center(k)
def response(q,k): return np.sum(np.asarray(q)*k,axis=-1)
def first_variation(q0,k0,dq,dk):return response(dq,k0)+response(q0,dk)
