# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
from pathlib import Path
from safetensors.torch import load_file
import torch
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
BASE = ROOT / 'outputs'
ARMS = {'S1': 1, 'S2': 2, 'S4': 4, 'S8': 8}
SEEDS = tuple(range(2026100101, 2026100109))
TIMES = (0, 16000, 64000, 256000)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
protocol = json.loads((OUT / 'RUN_PROTOCOL.json').read_text())
assert protocol['training_code_sha256'] == sha(ROOT / 'work/scientific/train_s1.py')
frozen_stream = json.loads((BASE / 'TRAINING_MANIFEST_HASHES.json').read_text())
entries = []
runs = []
failures = []
for seed in SEEDS:
    for arm, g in ARMS.items():
        folder = OUT / 'checkpoints' / f'{arm}_seed_{seed}'
        path = folder / 'RUN_RESULT.json'
        if not path.exists():
            failures.append(f'missing run result {arm} {seed}')
            continue
        r = json.loads(path.read_text())
        good = r['status'] == 'PASS' and r['arm'] == arm and (r['g'] == g) and (r['seed'] == seed) and (r['nonpadding_tokens_consumed'] == 256000) and (r['optimizer_steps'] == 1001) and r['only_router_trainable'] and r['finite_loss_gradient_and_router'] and math.isfinite(r['loss_first']) and math.isfinite(r['loss_last']) and math.isfinite(r['max_loss'])
        stream = frozen_stream['seeds'][str(seed)]
        good &= r['training_manifest_sha256'] == stream['manifest_sha256'] and r['training_tensor_sha256'] == stream['tensor_sha256']
        trace = folder / 'training_trace.jsonl'
        good &= trace.exists() and sha(trace) == r['trace_sha256']
        good &= sorted((int(t) for t in r['checkpoints'])) == list(TIMES)
        initial = r['frozen_tensor_hashes']
        for horizon in TIMES:
            item = r['checkpoints'].get(str(horizon))
            if item is None:
                good = False
                continue
            cpath = ROOT / item['router_checkpoint']
            if not cpath.exists():
                good = False
                continue
            actual = sha(cpath)
            good &= actual == item['router_checkpoint_sha256']
            good &= item['frozen_tensor_hashes'] == initial and item['router_finite']
            state = load_file(str(cpath))
            good &= list(state) == ['router.weight']
            good &= tuple(state['router.weight'].shape) == (4 * g, 512)
            good &= bool(torch.isfinite(state['router.weight']).all())
            entries.append({'arm': arm, 'seed': seed, 'horizon_nonpadding_tokens': horizon, 'router_checkpoint_sha256': actual, 'frozen_tensor_hashes': item['frozen_tensor_hashes']})
        if not good:
            failures.append(f'integrity failure {arm} {seed}')
        runs.append({'arm': arm, 'seed': seed, 'status': 'PASS' if good else 'FAIL', 'run_result_sha256': sha(path), 'trace_sha256': sha(trace) if trace.exists() else None, 'tokens': r['nonpadding_tokens_consumed'], 'steps': r['optimizer_steps'], 'training_loop_wall_seconds': r['training_loop_wall_seconds'], 'loss_first': r['loss_first'], 'loss_last': r['loss_last']})
extra = list((OUT / 'checkpoints').glob('*/RUN_RESULT.json'))
if len(extra) != 32:
    failures.append(f'run-result-count {len(extra)}')
entries.sort(key=lambda x: (x['seed'], x['arm'], x['horizon_nonpadding_tokens']))
aggregate = hashlib.sha256(b''.join(((json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n').encode() for x in entries))).hexdigest()
checkpoints = {'status': 'FROZEN' if not failures else 'FAIL', 'checkpoints': entries, 'checkpoint_count': len(entries), 'aggregate_sha256': aggregate, 'training_code_sha256': protocol['training_code_sha256']}
(OUT / 'CHECKPOINT_HASHES.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
integrity = {'status': 'PASS' if not failures else 'EXP_S1_BLOCKED_TRAINING_INTEGRITY', 'run_count': len(runs), 'expected_run_count': 32, 'checkpoints_verified': len(entries), 'expected_checkpoints': 128, 'total_nonpadding_tokens': sum((x['tokens'] for x in runs)), 'all_router_only_and_frozen_hashes_unchanged': not failures, 'checkpoint_hashes_json_sha256': sha(OUT / 'CHECKPOINT_HASHES.json'), 'failures': failures, 'runs': runs}
(OUT / 'TRAINING_INTEGRITY.json').write_text(json.dumps(integrity, indent=2) + '\n')
print(integrity['status'], 'runs', len(runs), 'checkpoints', len(entries), 'failures', failures, flush=True)
if failures:
    raise SystemExit(2)
