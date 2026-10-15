"""Verify the immutable public tree, source data and privacy/size boundaries."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
def verify(root):
    entries=json.loads((root/'PUBLIC_CHECKSUMS.json').read_text());checked=0;problems=[]
    for r in entries:
        p=root/r['path']
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        if h!=r['sha256']:raise ValueError('hash mismatch: '+r['path'])
        if p.stat().st_size!=r['bytes']:raise ValueError('size mismatch: '+r['path'])
        if p.stat().st_size>25*2**20:problems.append('large Git object: '+r['path'])
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:
                    if z[k].dtype.kind not in 'biufc':problems.append('non-numeric array: '+r['path']+'/'+k)
        elif p.suffix in ['.py','.md','.txt','.json','.csv','.yaml','.toml','.cff'] or p.name in ['LICENSE','.gitignore','.gitattributes']:
            text=p.read_text()
            patterns=[r'/(?:Users|home)/[A-Za-z0-9_]',r'(?i)bearer\s+[a-z0-9._-]{16,}',r'gh[pousr]_[A-Za-z0-9]{30,}',r'sk-[A-Za-z0-9]{20,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']
            if any(re.search(q,text) for q in patterns):problems.append('privacy/secret pattern: '+r['path'])
        checked+=1
    if problems:raise ValueError(problems)
    return {'files_verified':checked,'secret_path_scan':'PASS','binary_numeric_array_scan':'PASS','large_file_scan':'PASS'}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
