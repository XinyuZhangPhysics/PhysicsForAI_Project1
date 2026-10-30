import json,hashlib,collections,csv,shutil,sys,platform,importlib.metadata
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]; W=R.parent.parent; P=W/'work/Project1_NMIX_Unblock_Resume_Package_v1'; M=Path('<LOCAL_HOME>/AI_Workspace/ResearchModels/olmoe_5033B')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(n,d): (R/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 for folder,manifest in [(P,'PACKAGE_SHA256SUMS.txt'),(P/'frozen_package','SHA256SUMS.txt')]:
  for line in (folder/manifest).read_text().splitlines():
   h,n=line.split(maxsplit=1);assert sha(folder/n)==h,n
 shutil.copytree(P/'frozen_package',R/'frozen_package',dirs_exist_ok=True);shutil.copytree(P/'blocked_run_evidence',R/'blocked_run_evidence',dirs_exist_ok=True);shutil.copyfile(P/'BANK_PROVENANCE.md',R/'audit/BANK_PROVENANCE.md')
 cfg=json.loads((P/'frozen_package/protocol/nmix_frozen_protocol.json').read_text());shutil.copyfile(P/'frozen_package/protocol/nmix_frozen_protocol.json',R/'T0_FROZEN_CONFIG.json')
 b=P/'inputs/Formal_Bank_T_v1.jsonl';bb=b.read_bytes();records=[json.loads(x) for x in bb.splitlines()];assert len(bb)==1539879 and len(bb.splitlines())==len(records)==448 and sha(b)==cfg['bank']['expected_sha256']
 (R/'inputs').mkdir(exist_ok=True);shutil.copyfile(b,R/'inputs/Formal_Bank_T_v1.jsonl')
 domains=collections.Counter(r['domain'] for r in records);assert len(domains)==7 and set(domains.values())=={64};assert len({r['document_id'] for r in records})==448
 manifest=[];splits={}; tokens={}; dup={}
 for domain in sorted(domains):
  rows=sorted([(hashlib.sha256(r['document_id'].encode('utf-8')).hexdigest(),i,r) for i,r in enumerate(records) if r['domain']==domain])
  for rank,(key,i,r) in enumerate(rows):
   bank='A' if rank<32 else 'B'; ids=r['token_ids'][:96];assert len(ids)==96
   fullhash=hashlib.sha256(json.dumps(r['token_ids'],separators=(',',':')).encode()).hexdigest();tokenhash=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
   manifest.append([domain,r['document_id'],i,key,bank,tokenhash,fullhash]);splits[i]=bank;tokens[i]=ids;dup.setdefault(tokenhash,set()).add(bank)
 crossing=[k for k,v in dup.items() if len(v)>1];assert not crossing,'Cross-bank duplicate S96 sequences'
 with (R/'T0_BANK_SPLIT.tsv').open('w') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['domain','document_id','record_index','document_sha256','bank','S96_token_sha256','full_token_sha256']);w.writerows(manifest)
 put('T0_BANK_AUDIT.json',{'status':'VERIFIED','run_revision':4,'source':str(b),'bytes':len(bb),'records':len(records),'sha256':sha(b),'domains':dict(domains),'split':dict(collections.Counter(splits.values())),'split_rule':'SHA256 of UTF-8 immutable document_id, ascending per domain; first 32 A','sequence_rule':'Use first 96 archived token_ids for frozen S96; no retokenization. Archived records contain 256 IDs each.','cross_bank_duplicate_S96_count':len(crossing),'supplied_hash_identical_historical_input':True,'regenerated_or_substituted':False,'bank_B_scientific_evaluated':False})
 rng=np.random.Generator(np.random.PCG64(2026103001));a=rng.standard_normal((64,6));a-=a.mean(0);q,_=np.linalg.qr(a)
 for j in range(6):
  if q[np.argmax(abs(q[:,j])),j]<0:q[:,j]*=-1
 C=8*q;np.save(R/'T0_CONTRAST_BASIS.npy',C)
 rng=np.random.Generator(np.random.PCG64(2026103002));D=[]
 for j in range(12):
  x=rng.standard_normal(6);u=rng.standard_normal(6);D.append(np.r_[x/np.linalg.norm(x),u/np.linalg.norm(u)]/np.sqrt(2))
 np.save(R/'T0_DIRECTIONS.npy',D)
 assert np.max(abs(C.T@C/64-np.eye(6)))<1e-12
 modelaudit=[]
 for name,expected in cfg['model']['expected_sha256'].items():
  h=sha(M/name);modelaudit.append({'name':name,'path':str((M/name).resolve()),'sha256':h,'expected':expected,'matches':h==expected});print('model hash',name,h,flush=True);assert h==expected
 assert M.resolve().name==cfg['model']['revision_commit'];put('T0_MODEL_AUDIT.json',{'status':'VERIFIED','path':str(M),'revision':M.resolve().name,'files':modelaudit})
 versions={k:importlib.metadata.version(k) for k in ['torch','numpy','transformers','safetensors']};put('audit/ENVIRONMENT.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':versions})
 (R/'ENVIRONMENT.txt').write_text(json.dumps(json.loads((R/'audit/ENVIRONMENT.json').read_text()),indent=2)+'\n')
 print('PREFLIGHT PASS: verified package, exact bank, balanced split without cross-bank duplicates, model and independent arrays.',flush=True)
if __name__=='__main__':main()