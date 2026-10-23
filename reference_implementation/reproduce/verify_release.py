"""Verify source-release privacy and file-size boundaries."""
from pathlib import Path
import re

_TEXT_SUFFIXES={'.py','.md','.txt','.json','.csv','.yaml','.yml','.toml','.cff','.tsv'}
_SKIP_PARTS={'.git','.venv','results','__pycache__'}

def verify(root):
    root=Path(root)
    checked=0
    problems=[]
    patterns=[
        r'/(?:Users|home)/[A-Za-z0-9_]',
        r'(?i)bearer\s+[a-z0-9._-]{16,}',
        r'gh[pousr]_[A-Za-z0-9]{30,}',
        r'sk-[A-Za-z0-9]{20,}',
        r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',
    ]
    for p in root.rglob('*'):
        if not p.is_file() or any(part in _SKIP_PARTS for part in p.parts):
            continue
        rel=p.relative_to(root)
        if p.stat().st_size>25*2**20:
            problems.append('large Git object: '+str(rel))
        if p.suffix in _TEXT_SUFFIXES or p.name in {'LICENSE','.gitignore','.gitattributes'}:
            try:
                content=p.read_text()
            except UnicodeDecodeError:
                problems.append('non-UTF8 source/document: '+str(rel))
                continue
            if any(re.search(q,content) for q in patterns):
                problems.append('privacy/secret pattern: '+str(rel))
        checked+=1
    if problems:
        raise ValueError(problems)
    return {'source_files_verified':checked,'secret_path_scan':'PASS','large_file_scan':'PASS'}

if __name__=='__main__':
    import json
    print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
