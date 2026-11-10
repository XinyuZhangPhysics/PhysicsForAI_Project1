from pathlib import Path
import hashlib,json,os
import numpy as np

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
W=EXPERIMENT
COMMON=HERE.parents[1]/"common"
O=Path(os.environ.get("PROJECT1_NMIX_OUTPUT",EXPERIMENT/"outputs"))
O.mkdir(parents=True,exist_ok=True)
M=Path(os.environ["PROJECT1_OLMOE_MODEL"])
BANK=Path(os.environ["PROJECT1_NMIX_BANK_JSONL"])
R=COMMON
P=EXPERIMENT
CACHE_ROOT=O/"cache"
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
BASIS_SEEDS=(2026110902,2026110903,2026110904,2026110905)

def generate_basis(seed):
 rng=np.random.default_rng(seed)
 a=rng.standard_normal((64,6))
 a-=a.mean(0,keepdims=True)
 q,_=np.linalg.qr(a,mode="reduced")
 for j in range(6):
  i=int(np.argmax(np.abs(q[:,j])))
  if q[i,j]<0:q[:,j]*=-1
 return 8*q

def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def put(n,d):
 f=O/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")

CONFIG=json.loads((COMMON/"config.json").read_text())
