from pathlib import Path
import argparse, hashlib, json

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-dir',type=Path,required=True); args=ap.parse_args()
    d=args.result_dir; p=d/'generic_local_dynamics_manifest.json'; m=json.loads(p.read_text())
    if m.get('status')!='COMPLETE': raise SystemExit('Refusing to finalize non-COMPLETE result')
    vr=d/'RESULT_VERIFICATION.json'
    v=json.loads(vr.read_text())
    if v.get('status')!='PASS': raise SystemExit('RESULT_VERIFICATION is not PASS')
    env=d/'ENVIRONMENT_VERIFICATION.json'
    e=json.loads(env.read_text())
    if e.get('status')!='PASS': raise SystemExit('ENVIRONMENT_VERIFICATION is not PASS')
    m['postrun_recurrence_verification']='PASS'
    m['result_verification_sha256']=sha(vr)
    m['environment_verification_sha256']=sha(env)
    outputs=['generic_local_dynamics_long.csv','generic_local_dynamics_endpoints.csv','generic_local_dynamics_operator.npz','generic_local_dynamics_summary.json','Fig_generic_local_dynamics_diagnostic.pdf','SMOKE_TEST.json','RESULT_VERIFICATION.json','ENVIRONMENT_VERIFICATION.json']
    m['output_sha256']={name:sha(d/name) for name in outputs}
    p.write_text(json.dumps(m,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('FINAL_MANIFEST PASS')

if __name__=='__main__': main()
