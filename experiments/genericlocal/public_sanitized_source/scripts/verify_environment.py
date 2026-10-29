from pathlib import Path
import argparse, json, platform, sys

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    record={'status':'STOP','python':sys.executable,'python_version':sys.version,'platform':platform.platform()}
    try:
        import numpy, torch, transformers
        record.update(numpy=numpy.__version__,torch=torch.__version__,transformers=transformers.__version__)
        required={'numpy':'2.5.3','torch':'2.14.0','transformers':'5.17.0'}
        problems=[]
        if sys.version_info[:2] != (3,12): problems.append(f'Python {sys.version_info.major}.{sys.version_info.minor} != 3.12')
        for k,v in required.items():
            if record[k] != v: problems.append(f'{k} {record[k]} != {v}')
        try:
            import matplotlib
            record['matplotlib']=matplotlib.__version__
        except Exception as e:
            problems.append('matplotlib import failed: '+str(e))
        if problems:
            record['problems']=problems
        else:
            record['status']='PASS'; record['required_versions']=required; record['python_major_minor']='3.12'
    except Exception as e:
        record['problems']=[f'environment import failure: {type(e).__name__}: {e}']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(record,indent=2,sort_keys=True))
    if record['status']!='PASS': raise SystemExit(3)

if __name__=='__main__': main()
