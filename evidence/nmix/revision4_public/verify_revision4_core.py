from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def mat(name):
    return np.loadtxt(DATA/name, delimiter="\t", dtype=np.float64)

Hraw=mat("T2_HESSIAN_A_RAW.tsv")
H=mat("T2_HESSIAN_A_SYM.tsv")
assert np.array_equal(H,(Hraw+Hraw.T)/2)
Hd=H.copy(); Hd[:6,6:]=0; Hd[6:,:6]=0

with (DATA/"B_CONTROL_SPACE_RAW.tsv").open(newline="") as f:
    raw=list(csv.DictReader(f,delimiter="\t"))
states=np.array([[float(r[f"state_{j}"]) for j in range(12)] for r in raw])
responses=np.array([[float(r[f"response_{j}"]) for j in range(12)] for r in raw])
with (DATA/"T3_BANK_B_STATE_METRICS.csv").open(newline="") as f:
    rr=list(csv.DictReader(f))
assert len(rr)==72==len(raw)

def cosine(a,b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))

for i,row in enumerate(rr):
    r=responses[i]; z=states[i]
    ef=np.linalg.norm(r-H@z); ed=np.linalg.norm(r-Hd@z)
    assert np.isclose(ef,float(row["full_error"]),rtol=1e-13)
    assert np.isclose(ed,float(row["no_mixed_error"]),rtol=1e-13)
    assert np.isclose(np.linalg.norm(r),float(row["response_norm"]),rtol=1e-13)
    assert np.isclose(cosine(r,H@z),float(row["full_cosine"]),rtol=1e-13)

summary=json.loads((DATA/"T3_BANK_B_PRIMARY_SUMMARY.json").read_text())
p=summary["primary"]
ids=[i for i,row in enumerate(rr) if float(row["amplitude"])==0.02]
R=np.sqrt(sum(float(rr[i]["full_error"])**2 for i in ids)/sum(float(rr[i]["no_mixed_error"])**2 for i in ids))
assert np.isclose(R,p["R"],rtol=1e-13)

rng=np.random.Generator(np.random.PCG64(2026103003))
boot=np.empty((10000,2))
for k in range(10000):
    chosen=[ids[2*i+s] for i in rng.integers(0,12,12) for s in range(2)]
    boot[k,0]=np.sqrt(sum(float(rr[i]["full_error"])**2 for i in chosen)/sum(float(rr[i]["no_mixed_error"])**2 for i in chosen))
    boot[k,1]=np.median([float(rr[i]["full_cosine"]) for i in chosen])
ciR=np.quantile(boot[:,0],[0.025,0.975]).tolist()
ciC=np.quantile(boot[:,1],[0.025,0.975]).tolist()
assert np.allclose(ciR,p["R_bootstrap_ci95"],rtol=1e-13,atol=1e-15)
assert np.allclose(ciC,p["median_C_full_bootstrap_ci95"],rtol=1e-13,atol=1e-15)

turn=np.array([float(rr[i]["support_turnover_fraction"]) for i in ids])
assert np.isclose(np.median(turn),p["median_support_turnover"],rtol=1e-13)
assert np.isclose(np.quantile(turn,0.95),p["p95_support_turnover"],rtol=1e-13)

adj=json.loads((DATA/"NMIX_ADJUDICATION.json").read_text())
assert adj["final_status"]=="GRADE_A_DELETE_NATIVE_QUALIFICATION"
assert all(x["pass"] for x in adj["grade_A_checks"].values())
assert not adj["grade_A_plus_checks"]["median_support_turnover"]["pass"]
assert not adj["grade_A_plus_checks"]["p95_support_turnover"]["pass"]

print("NMIX revision-4 public core: PASS")
print("E_full =",p["E_full"])
print("E_no_mixed =",p["E_diag"])
print("R =",p["R"])
print("R 95% paired-direction bootstrap CI =",ciR)
print("median cosine =",p["median_C_full"])
print("support turnover median/p95 =",p["median_support_turnover"],p["p95_support_turnover"])
print("Grade A PASS; Grade A+ fails only the two support-turnover limits")