from pathlib import Path
import csv
import numpy as np

R=Path(__file__).resolve().parent
H=np.loadtxt(R/"H_src.tsv",delimiter="\t")
with (R/"heldout_responses_a002.tsv").open(newline="") as f:
    rows=list(csv.DictReader(f,delimiter="\t"))
Z=np.array([[float(r[f"state_{j}"]) for j in range(12)] for r in rows])
Y=np.array([[float(r[f"response_{j}"]) for j in range(12)] for r in rows])
assert Z.shape==Y.shape==(24,12)
def error(K):
    return np.linalg.norm(Y-Z@K.T)/np.linalg.norm(Y)
def bd(K):
    Q=K.copy();Q[:6,6:]=0;Q[6:,:6]=0;return Q
ef,eb=error(H),error(bd(H))
assert np.isclose(ef,0.10345120235788188,atol=3e-12)
assert np.isclose(eb,0.5867084710549377,atol=3e-12)
A=np.linalg.lstsq(Z,Y,rcond=None)[0]
es=np.linalg.norm(Y-Z@A)/np.linalg.norm(Y)
assert np.isclose(es,0.03885301385,atol=3e-10)
M=H[6:,:6]
U,S,Vt=np.linalg.svd(M,full_matrices=False)
errs=[]
for k in range(7):
    Mk=(U[:,:k]*S[:k])@Vt[:k,:] if k else np.zeros_like(M)
    K=H.copy();K[6:,:6]=Mk;K[:6,6:]=Mk.T
    errs.append(error(K))
rounded=np.round(errs,3)
assert np.array_equal(rounded,np.array([0.587,0.439,0.334,0.250,0.195,0.155,0.103]))
dpart=(np.sum(S*S)**2)/np.sum(S**4)
assert np.isclose(dpart,3.4899845217893466,atol=1e-12)
def invsqrt(A):
    w,V=np.linalg.eigh(A)
    assert np.min(w)>0
    return (V*(w**-0.5))@V.T
W=invsqrt(H[6:,6:])@M@invsqrt(H[:6,:6])
kappa=np.linalg.svd(W,compute_uv=False)[0]
assert np.isclose(kappa,0.759,atol=5e-4)
print("E_full",ef)
print("E_bd",eb)
print("E_secant",es)
print("rank_k",errs)
print("d_part_mix",dpart)
print("kappa",kappa)
print("public v1 primary OLMoE verification PASS")