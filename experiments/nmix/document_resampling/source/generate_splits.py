import argparse,csv
from pathlib import Path
import numpy as np
SEEDS=(2026110911,2026110912,2026110913,2026110914)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--manifest",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
 rows=list(csv.DictReader(open(args.manifest),delimiter="\t"));domains=sorted({r["domain"] for r in rows});out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 for j,seed in enumerate(SEEDS,1):
  rng=np.random.default_rng(seed);new=[]
  for dom in domains:
   rr=sorted([dict(r) for r in rows if r["domain"]==dom],key=lambda r:int(r["record_index"]));p=rng.permutation(len(rr));aset={int(p[k]) for k in range(32)}
   for k,r in enumerate(rr):r["bank"]="A" if k in aset else "B";new.append(r)
  new=sorted(new,key=lambda r:(r["domain"],int(r["record_index"])))
  with (out/f"split_{j:02}.tsv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(new)
if __name__=="__main__":main()
