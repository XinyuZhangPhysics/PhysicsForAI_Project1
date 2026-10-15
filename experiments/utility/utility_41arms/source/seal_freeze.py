# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Seal all scientific identities before the first Q3 model computation."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import inspect, subprocess
import numpy as np
import torch, transformers
from firewall import *

def main():
    assert read(BUNDLE / 'entry_gate_audit.json')['status'] == 'PASS'
    assert not (BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json').exists(), 'FREEZE_ALREADY_PRESENT'
    p = subprocess.run([sys.executable, str(ROOT / 'test_firewall.py')], capture_output=True, text=True)
    (BUNDLE / 'targeted_firewall_tests.log').write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError('FIREWALL_TEST_FAILURE')
    metadata = [json.loads(line) for line in (ROOT / 'metadata/candidate_frame_metadata.jsonl').open()]
    index = build_index(FRAME, metadata, BUNDLE / 'blind_byte_index.json')
    audit = read(BUNDLE / 'bank_overlap_audit.json')
    parts = {}
    for bank, spec in audit['manifests'].items():
        parts[bank] = [json.loads(line) for line in (BUNDLE / spec['file']).open()]
    fw = Firewall(FRAME, index, parts, BUNDLE, BUNDLE / 'access_events.jsonl', 'PRE_SCIENCE_NO_SEAL')
    denied = []
    for bank, method in [('HELDOUT', fw.load_heldout), ('RESERVE', fw.load_adapt), ('RESERVE', fw.load_heldout)]:
        try:
            method(parts[bank][0]['document_id'], 'CANDIDATE')
        except AccessDenied as e:
            denied.append(dict(bank=bank, reason=str(e)))
        else:
            raise RuntimeError('REAL_BANK_NEGATIVE_ACCESS_TEST_FAILED')
    assert len(denied) == 3 and (not any(fw.decoded_counts.values()))
    write(BUNDLE / 'access_firewall_preflight_audit.json', dict(status='PASS', synthetic_tests=7, real_bank_denied_tests=denied, real_payload_decodes=0, blind_index_records=len(index), index_source_sha256=sha(FRAME)))
    event('BLIND_INDEX_AND_FIREWALL_PASS', index_sha256=sha(BUNDLE / 'blind_byte_index.json'), real_payload_decodes=0)
    source = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    from transformers.models.olmoe import modeling_olmoe
    source['installed_transformers_modeling_olmoe'] = info(inspect.getfile(modeling_olmoe))
    write(BUNDLE / 'prospective_source_provenance.json', dict(fresh_Q3_implementation=True, prior_Q3_runner_state_imported=False, prior_Q3_model_or_alpha_state_imported=False, source_hashes=source, source_parent=str(ROOT)))
    entry = read(BUNDLE / 'entry_gate_audit.json')
    freeze = dict(version='v176', scope='Q3_ONLY', time_UTC=now(), model_repository='allenai/OLMoE-1B-7B-0924', model_sha=MODEL_SHA, layers=[7, 15], scientific_protocol_sha256=sha(ROOT / 'authorities/PROJECT1_V173_PROTOCOL_FREEZE.md'), amendment_sha256=sha(ROOT / 'PROJECT1_V176_ACCESS_AMENDMENT.md'), inherited_identity_checks=entry['checks'], Bank_V_payload_root_sha256=entry['bank_v_payload_root_sha256'], manifests=audit['manifests'], arms=ARMS, blind_index_sha256=sha(BUNDLE / 'blind_byte_index.json'), prospective_source_sha256=sha(BUNDLE / 'prospective_source_provenance.json'), optimizer=dict(name='Adam', lr=0.0005, betas=[0.9, 0.999], eps=1e-08, weight_decay=0, microbatch_documents=1, gradient_accumulation_documents=8, epochs=2, initial_alphas=[0, 0], projection=[-0.02, 0.02], projection_after_every_step=True, steps_per_arm=56), document_order='round-robin in protocol domain order; token-array SHA ascending within each domain; same order in each epoch', loss='ordinary next-token cross-entropy NLL, all 255 predicted tokens in each 256-token document, no added tokens or masks', intervention='add alpha_L times exact FP32-converted router-bias block to FP32-converted native BF16 logits at gate output before native FP32 softmax and Top-k', base_weights='frozen BF16, eager attention, no quantization', trainable_parameters='only two FP32 real scalars', bootstrap=dict(replicates=10000, seed_string='PROJECT1-Q3-BOOTSTRAP-v173', seed_conversion='first 8 SHA256 UTF8 bytes interpreted unsigned little endian', generator='NumPy PCG64', paired=True, stratified_domains=7, documents_per_domain=64, CI='95% percentile', quantile_method='linear'), endpoint='D = U_CANDIDATE - mean(U_RANDOM_000:038)', runtime=read(BUNDLE / 'runtime_identity.json'), historical_disclosures=HISTORY, **HISTORY, Q3_RESERVE_USED_FOR_SCIENCE=False, Q3_HELDOUT_MODEL_ACCESS_BEFORE_ARM_SEAL=False, POST_Q3_EXPERIMENTS_NOT_EXECUTED=True, scientific_outcomes_generated_at_seal=0, Q3_model_forwards_at_seal=0)
    write(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json', freeze)
    (BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.sha256').write_text(sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json') + '  Q3_SCIENTIFIC_FREEZE_v176.json\n')
    for p in [BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json', BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.sha256']:
        p.chmod(292)
    event('SCIENTIFIC_FREEZE_SEALED', freeze_sha256=sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'), Q3_model_forwards=0)
    print('SCIENTIFIC FREEZE SEALED; firewall 7 synthetic + 3 real-bank denial tests PASS; payload decodes 0', flush=True)
if __name__ == '__main__':
    main()
