"""Stationary mixed-curvature block and exact discrete growth."""
import numpy as np
class CrossOperator:
    def __init__(self,h,phi,gradient,coefficient=.25):
        self.h=np.asarray(h);self.phi=np.asarray(phi);self.gradient=np.asarray(gradient);self.coefficient=coefficient
    def c(self,m):return -self.coefficient*self.h.T@np.sum(self.gradient*(self.phi@np.asarray(m).T),axis=1)
    def ct(self,u):return -self.coefficient*self.gradient.T@((self.h@u)[:,None]*self.phi)
    def matrix(self):
        return np.stack([self.ct(e).ravel() for e in np.eye(self.h.shape[1])])
def growth_rate(mu,kappa,lambda_,eta_m,eta_u):
    a=eta_m*mu;b=eta_u*lambda_
    sigma=-(a+b)/2+np.sqrt((a-b)**2+4*eta_m*eta_u*kappa*kappa)/2
    return np.log1p(sigma)
def zeta(eta_m,eta_u,lambda_d,mu):return eta_u/eta_m*lambda_d/mu
def initialize(u1,v1,mu,kappa,lambda_,eta_m,eta_u,amplitude,sign=1):
    a=np.array([[-eta_m*mu,eta_m*kappa],[eta_u*kappa,-eta_u*lambda_]])
    values,vectors=np.linalg.eig(a);e=vectors[:,np.argmax(values.real)].real
    if e[0]<0:e=-e
    e/=np.linalg.norm(e)
    return sign*amplitude*e[0]*v1,sign*amplitude*e[1]*u1
def relative_l2(observed,predicted):return np.linalg.norm(np.asarray(observed)-predicted)/np.linalg.norm(predicted)
