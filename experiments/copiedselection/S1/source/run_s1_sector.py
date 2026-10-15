# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).parent))
from s1_impl import ARMS, parent_vectors
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SEED = 2026100101
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')

def project(v, g):
    pa = v.mean(axis=2, keepdims=True) - v.mean(axis=(1, 2), keepdims=True)
    pc = v.mean(axis=1, keepdims=True) - v.mean(axis=(1, 2), keepdims=True)
    pm = v - pa - pc - v.mean(axis=(1, 2), keepdims=True)
    return (pa, pc, pm)

def main():
    data = torch.load(ROOT / 'work/s1_calibration_bank.pt', weights_only=True, map_location='cpu')
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / 'work/model'), dtype=torch.float32, local_files_only=True).to('mps').eval()
    dense = model.gpt_neox.layers[3].mlp
    inputs = []

    def capture(_module, args):
        inputs.append(args[0].detach())
    hook = dense.register_forward_pre_hook(capture)
    with torch.inference_mode():
        for i in range(0, 2048, 8):
            mask = data['attention_mask'][i:i + 8].to('mps')
            model.gpt_neox(input_ids=data['input_ids'][i:i + 8].to('mps'), attention_mask=mask, use_cache=False)
            h = inputs.pop()
            last = mask.sum(dim=1).long() - 1
            inputs.append(h[torch.arange(len(h), device='mps'), last].cpu())
    hook.remove()
    x = torch.cat(inputs, dim=0)
    with torch.inference_mode():
        activation = dense.act(dense.dense_h_to_4h(x.to('mps'))).cpu().numpy().astype(np.float64)
    x = x.cpu().numpy().astype(np.float64)
    assert x.shape == (2048, 512)
    torch.save({'h': torch.from_numpy(x.astype(np.float32))}, ROOT / 'work/s1_calibration_representation.pt')
    bank_rows = [json.loads(line) for line in (OUT / 'calibration_bank_manifest.jsonl').open()]
    rowmap = {row['train_row_id']: i for i, row in enumerate(bank_rows)}
    pair_path = OUT / 'calibration_pair_manifest.jsonl'
    pair_rows = [json.loads(line) for line in pair_path.open()]
    assert len(pair_rows) == 12288
    src = np.asarray([rowmap[r['source_train_row_id']] for r in pair_rows])
    tgt = np.asarray([rowmap[r['target_train_row_id']] for r in pair_rows])
    results = {}
    zero_powers = []
    for arm, g in ARMS.items():
        m = 2048 // g
        parent = parent_vectors(SEED, torch.float64).numpy()
        router = np.repeat(parent, g, axis=0)
        logits = x @ router.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        q = (probs[src] + probs[tgt]) / 2
        q = q.reshape(-1, 4, g)
        child = np.empty((len(src), g), dtype=np.float64)
        for j in range(g):
            a = activation[:, j * m:(j + 1) * m]
            child[:, j] = (a[src] * a[tgt]).mean(axis=1)
        k = np.repeat(child[:, None, :], 4, axis=1)
        qa, qc, qm = project(q, g)
        ka, kc, km = project(k, g)
        reconstruction_q = float(np.max(np.abs(q - q.mean(axis=(1, 2), keepdims=True) - qa - qc - qm)))
        reconstruction_k = float(np.max(np.abs(k - k.mean(axis=(1, 2), keepdims=True) - ka - kc - km)))
        B = np.sum(qc * kc, axis=(1, 2))
        power = float(np.mean(B * B))
        q_c_power = float(np.mean(np.square(qc).sum(axis=(1, 2))))
        forbidden_k_power = float(np.mean(np.square(ka).sum(axis=(1, 2)) + np.square(km).sum(axis=(1, 2))))
        residual_power = max(reconstruction_q ** 2, reconstruction_k ** 2)
        zero_powers.extend((power, q_c_power, forbidden_k_power, residual_power))
        arbitrary_router_power = None
        if g == 1:
            random_router = np.random.default_rng(2026100197).normal(size=(4, 512))
            other = x @ random_router.T
            other -= other.max(axis=1, keepdims=True)
            other = np.exp(other)
            other /= other.sum(axis=1, keepdims=True)
            q_other = (other[src] + other[tgt]) / 2
            k_centered = k.reshape(-1, 4) - k.reshape(-1, 4).mean(axis=1, keepdims=True)
            arbitrary_router_power = float(np.mean(np.square(np.sum(q_other * k_centered, axis=1))))
            zero_powers.append(arbitrary_router_power)
        results[arm] = {'g': g, 'H': 4 * g, 'm': m, 'C_dimension': g - 1 if g > 1 else 0, 'M_dimension': 3 * (g - 1) if g > 1 else 0, 'q_C_power': q_c_power, 'kappa_forbidden_A_M_power': forbidden_k_power, 'B0_power': power, 'reconstruction_max_abs': max(reconstruction_q, reconstruction_k), 'kappa_C_power': float(np.mean(np.square(kc).sum(axis=(1, 2)))), 'S1_arbitrary_router_B_power': arbitrary_router_power, 'pass': q_c_power <= 1e-20 and forbidden_k_power <= 1e-20 and (power <= 1e-20) and (g > 1 or arbitrary_router_power <= 1e-20) and (g == 1 or float(np.mean(np.square(kc).sum(axis=(1, 2)))) > 0)}
        print(arm, results[arm], flush=True)
    largest = max(zero_powers)
    floor = max(1e-20, 1000 * largest)
    rule = {'source': 'Full 2048-document calibration bank; layer-4 input at final nonpadding token; frozen 12 ordered domain-pair coordinates with 1024 endpoint pairs each', 'largest_numerical_zero_power': largest, 'multiplier': 1000, 'absolute_floor': 1e-20, 'formal_H1_rule': 'All S1 audit B power coordinates at every registered checkpoint and seed must be <= numerical_zero_power_threshold.', 'numerical_zero_power_threshold': floor, 'strictly_at_least_100x_largest_zero_power': floor >= 100 * largest, 'frozen_before_scientific_training': True}
    (OUT / 'ZERO_FLOOR_RULE.json').write_text(json.dumps(rule, indent=2) + '\n')
    result = {'seed': SEED, 'calibration_sequences': 2048, 'ordered_pair_coordinates': 12, 'pairs_per_coordinate': 1024, 'pair_manifest_sha256': hashlib.sha256(pair_path.read_bytes()).hexdigest(), 'representation_tensor_sha256': hashlib.sha256((ROOT / 'work/s1_calibration_representation.pt').read_bytes()).hexdigest(), 'arms': results, 'pass': all((r['pass'] for r in results.values()))}
    (OUT / 'SECTOR_CONTROLS.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SECTOR PASS', result['pass'], 'floor', floor, flush=True)
if __name__ == '__main__':
    main()
