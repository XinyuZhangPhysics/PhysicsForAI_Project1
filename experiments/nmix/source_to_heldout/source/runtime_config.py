from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
R=HERE.parents[1]/"common"
O=Path(os.environ.get("PROJECT1_NMIX_OUTPUT",EXPERIMENT/"outputs"));O.mkdir(parents=True,exist_ok=True)
M=Path(os.environ["PROJECT1_OLMOE_MODEL"])
BANK=Path(os.environ["PROJECT1_NMIX_BANK_JSONL"])
P=EXPERIMENT
CACHE_ROOT=Path(os.environ.get("PROJECT1_NMIX_CACHE_ROOT",O/"cache"));CACHE_ROOT.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((R/"config.json").read_text())
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def put(n,d):
 f=O/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
