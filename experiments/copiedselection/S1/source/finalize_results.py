# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
BASE = ROOT / 'outputs'
WORK = ROOT / 'work/scientific'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(name):
    return json.loads((OUT / name).read_text())
pre = json.loads((BASE / 'EXP_S1_PREFLIGHT.json').read_text())
integrity = load('TRAINING_INTEGRITY.json')
checkpoint = load('CHECKPOINT_HASHES.json')
cal = load('CALIBRATION_FREEZE.json')
opening = load('AUDIT_OPENING_RECORD.json')
primary = load('AUDIT_PRIMARY_64K.json')
reg = load('PRIMARY_REGRESSIONS.json')
boot = load('BOOTSTRAP.json')
hyp = load('FORMAL_HYPOTHESES.json')
formal = load('FORMAL_LABEL.json')
freeze = load('PRIMARY_FREEZE.json')
secondary = load('SECONDARY_TEMPORAL.json')
assert integrity['status'] == 'PASS' and integrity['run_count'] == 32
assert checkpoint['checkpoint_count'] == 128
assert cal['status'] == 'CALIBRATION_FROZEN'
assert opening['status'] == 'AUDIT_OPENED_AFTER_FROZEN_PREREQUISITES'
assert freeze['central_label'] == formal['label'] == secondary['primary_label_unchanged']
for name, expected in freeze['files_sha256'].items():
    assert sha(OUT / name) == expected
assert integrity['total_nonpadding_tokens'] == 8192000
assert all((primary['resolved_count_by_arm'][arm] == 8 for arm in ('S2', 'S4', 'S8')))
slopes = formal['central_slopes']
ci = formal['ci95']
beta_k = reg['descriptive_beta_K']
factor_slope_error = abs(slopes['beta_B'] - (slopes['beta_Q'] + beta_k + slopes['beta_Theta']))
assert factor_slope_error < 1e-10
all_h1 = primary['S1_H1_rows']
assert len(all_h1) == 32 and all((x['all_12_below_floor'] for x in all_h1))
assert boot['replicates'] == 20000 and boot['seed'] == 2026100199
assert sha(OUT / 'BOOTSTRAP_INDICES.npy') == boot['indices_sha256']
code_files = ('train_s1.py', 'freeze_training.py', 'response_s1.py', 'evaluate_s1.py', 'materialize_audit.py', 'freeze_calibration.py', 'open_audit.py', 'verify_provenance.py')
code_hashes = {name: sha(WORK / name) for name in code_files}
assert code_hashes['response_s1.py'] == opening['analysis_code_sha256']['response_s1.py']
assert code_hashes['evaluate_s1.py'] == opening['analysis_code_sha256']['evaluate_s1.py']
assert code_hashes['materialize_audit.py'] == opening['analysis_code_sha256']['materialize_audit.py']
protocol_dir = WORK / 'PhysicsForAI_EXP_S1_Scientific'
protocol_hashes = {p.name: sha(p) for p in protocol_dir.iterdir() if p.is_file()}
provenance = {'status': 'FROZEN', 'timestamp_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'preflight_json_sha256': sha(BASE / 'EXP_S1_PREFLIGHT.json'), 'scientific_archive_sha256': sha(Path(_p1_path('path_b2b55db8f49e'))), 'model_revision': pre['model_revision'], 'dataset_revision': pre['dataset_revision'], 'protocol_document_sha256': protocol_hashes, 'scientific_code_sha256': code_hashes, 'preflight_code_sha256': pre['code_hashes'], 'run_protocol_sha256': sha(OUT / 'RUN_PROTOCOL.json'), 'checkpoint_aggregate_sha256': checkpoint['aggregate_sha256'], 'checkpoint_hashes_file_sha256': sha(OUT / 'CHECKPOINT_HASHES.json'), 'training_integrity_sha256': sha(OUT / 'TRAINING_INTEGRITY.json'), 'calibration_freeze_sha256': sha(OUT / 'CALIBRATION_FREEZE.json'), 'audit_opening_record_sha256': sha(OUT / 'AUDIT_OPENING_RECORD.json'), 'primary_freeze_sha256': sha(OUT / 'PRIMARY_FREEZE.json'), 'formal_label_sha256': sha(OUT / 'FORMAL_LABEL.json'), 'secondary_temporal_sha256': sha(OUT / 'SECONDARY_TEMPORAL.json'), 'audit_opened_after_all_frozen_prerequisites': True, 'no_seed_replacement_hyperparameter_tuning_or_primary_horizon_change': True}
(OUT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
results = {'status': 'COMPLETE', 'formal_label': formal['label'], 'formal_label_final': True, 'H1': hyp['H1'], 'H2': hyp['H2'], 'H3': hyp['H3'], 'H4': hyp['H4'], 'resolved_requirement': hyp['resolved_requirement'], 'resolved_counts_by_arm': primary['resolved_count_by_arm'], 'primary_horizon_nonpadding_tokens': 64000, 'central_slopes': slopes, 'ci95': ci, 'descriptive_K_C_0_by_arm': reg['descriptive_K_C_0_by_arm'], 'descriptive_beta_K': beta_k, 'factorized_slope_identity_abs_error': factor_slope_error, 'S1_max_audit_B_coordinate_power': max((x['max_B_coordinate_power'] for x in all_h1)), 'H1_zero_floor': 1e-20, 'secondary_temporal': secondary['horizons'], 'training_runs': 32, 'checkpoint_count': 128, 'total_nonpadding_training_tokens': 8192000, 'bootstrap_replicates': 20000, 'bootstrap_seed': 2026100199, 'formal_label_sha256': sha(OUT / 'FORMAL_LABEL.json'), 'primary_freeze_sha256': sha(OUT / 'PRIMARY_FREEZE.json'), 'provenance_sha256': sha(OUT / 'PROVENANCE.json')}
(OUT / 'EXP_S1_RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')

def fmt(v):
    return f'{v:.4f}'
report = f"# EXP-S1 scientific results\n\n**Formal label: `{formal['label']}`**  \n**Primary horizon:** 64,000 non-padding tokens.  \n**Execution:** 32/32 paired-seed router-only runs completed; 128/128 registered checkpoints verified; total 8,192,000 non-padding training tokens. No seed was replaced, no hyperparameter was tuned, and the primary horizon stayed at 64k.\n\n## Formal hypotheses\n\n| Hypothesis | Rule | Result |\n|---|---|---|\n| H1, S1 structural null | Every audit B-coordinate power at all four horizons and eight seeds ≤ 1e-20 | **{hyp['H1']}**; maximum {results['S1_max_audit_B_coordinate_power']:.3g} |\n| H2, alignment dilution | 95% upper bound of βΘ < 0 | **{hyp['H2']}** |\n| H3, router leakage amplification | 95% lower bound of βQ > 0 | **{hyp['H3']}** |\n| H4, net private-response growth | 95% lower bound of βB > 0 | **{hyp['H4']}** |\n| Resolved positive response | At least 7/8 seeds for each of S2/S4/S8 | **{hyp['resolved_requirement']}**; 8/8 for each arm |\n\n## Primary 64k slopes\n\n| Quantity | Paired-seed slope in log g | 95% whole-seed bootstrap interval |\n|---|---:|---:|\n| βQ | {fmt(slopes['beta_Q'])} | [{fmt(ci['beta_Q'][0])}, {fmt(ci['beta_Q'][1])}] |\n| βΘ | {fmt(slopes['beta_Theta'])} | [{fmt(ci['beta_Theta'][0])}, {fmt(ci['beta_Theta'][1])}] |\n| βB | {fmt(slopes['beta_B'])} | [{fmt(ci['beta_B'][0])}, {fmt(ci['beta_B'][1])}] |\n\nThe seed fixed-effect regression uses only S2/S4/S8 at 64k. The 20,000 bootstrap replicates resample whole paired seed triplets with seed 2026100199. No unresolved value was regularized into a log regression.\n\nFrozen initial child-sector geometry is K_C⁽⁰⁾ = {reg['descriptive_K_C_0_by_arm']['S2']:.6g} (S2), {reg['descriptive_K_C_0_by_arm']['S4']:.6g} (S4), and {reg['descriptive_K_C_0_by_arm']['S8']:.6g} (S8), with descriptive βK = {fmt(beta_k)}. This is not a formal PASS criterion. The factorized slope identity βB = βQ + βK + βΘ holds to absolute error {factor_slope_error:.3g}.\n\n## Secondary timing\n\n| Horizon | βQ | βΘ | βB |\n|---|---:|---:|---:|\n| 16k | {fmt(secondary['horizons']['16000']['beta_Q'])} | {fmt(secondary['horizons']['16000']['beta_Theta'])} | {fmt(secondary['horizons']['16000']['beta_B'])} |\n| 256k | {fmt(secondary['horizons']['256000']['beta_Q'])} | {fmt(secondary['horizons']['256000']['beta_Theta'])} | {fmt(secondary['horizons']['256000']['beta_B'])} |\n\nThese temporal slopes are descriptive. They were computed only after the 64k formal label and its hashes were frozen; they did not change the label.\n\n## Integrity and audit order\n\nThe exact Pythia-70M and SlimPajama commits, 48 dataset shards, fresh banks, eight training streams, router initialization, and preflight code hashes were verified before execution. Only the layer-4 router changed; frozen expert, base-model, and buffer hashes matched at all checkpoints. All losses and gradients were finite.\n\nCheckpoint hashes, training integrity, and calibration response controls were frozen before `AUDIT_OPENING_RECORD.json` was written. The audit representation and scientific response observables were materialized only after that record. The response analysis uses GT2-E4's pair-product softmax router normalization and the registered 12 ordered domain-pair coordinates. The final label is fixed; no repair run or follow-on experiment was started.\n\nSee `EXP_S1_RESULTS.json` for machine-readable values, `PROVENANCE.json` for hashes, and the individual audit response files and checkpoints for the complete record.\n"
(OUT / 'EXP_S1_REPORT.md').write_text(report)
print(formal['label'], 'report complete', flush=True)
