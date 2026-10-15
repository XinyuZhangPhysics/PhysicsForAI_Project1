# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).parent))
from s1_impl import ARMS, SoftRouterMLP, parent_vectors, three_functions
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'work/model'
OUT = ROOT / 'outputs'
SEED = 2026100101

def rel(a, b):
    return float((a.double() - b.double()).norm().item() / max(a.double().norm().item(), 1e-30))

def model_copy(dtype, device):
    return AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=dtype, local_files_only=True).to(device).eval()

def cpu_gate(data):
    base = model_copy(torch.float32, 'mps')
    dense = base.gpt_neox.layers[3].mlp
    values = []
    masks = []

    def capture(_module, inputs):
        values.append(inputs[0].detach().cpu())
    hook = dense.register_forward_pre_hook(capture)
    with torch.inference_mode():
        for i in range(0, 256, 4):
            masks.append(data['attention_mask'][i:i + 4].bool())
            base(input_ids=data['input_ids'][i:i + 4].to('mps'), attention_mask=data['attention_mask'][i:i + 4].to('mps'), use_cache=False)
    hook.remove()
    x = torch.cat([v[mask] for v, mask in zip(values, masks)], dim=0).double()
    assert tuple(x.shape) == (59707, 512)
    dense64 = copy.deepcopy(dense).cpu().double()
    out = {}
    with torch.inference_mode():
        for arm, g in ARMS.items():
            moe = SoftRouterMLP(dense64, g, parent_vectors(SEED, torch.float64)).double()
            accum = {'bd_diff': 0.0, 'bd_ref': 0.0, 'mb_diff': 0.0, 'mb_ref': 0.0}
            for chunk in x.split(512):
                a, b, c = three_functions(dense64, moe, chunk)
                accum['bd_diff'] += float((b - a).square().sum())
                accum['bd_ref'] += float(a.square().sum())
                accum['mb_diff'] += float((c - b).square().sum())
                accum['mb_ref'] += float(b.square().sum())
            bd = math.sqrt(accum['bd_diff'] / accum['bd_ref'])
            mb = math.sqrt(accum['mb_diff'] / accum['mb_ref'])
            out[arm] = {'sequences': 256, 'valid_tokens': len(x), 'relF_block_dense': bd, 'relF_moe_block': mb, 'pass': bd <= 1e-09 and mb <= 1e-09}
            print('CPU', arm, out[arm], flush=True)
    del base
    return out

def mps_gate(data):
    assert torch.backends.mps.is_available()
    base = model_copy(torch.float32, 'mps')
    dense = base.gpt_neox.layers[3].mlp
    results = {}
    for arm, g in ARMS.items():
        moe = SoftRouterMLP(dense, g, parent_vectors(SEED, torch.float32).to('mps')).to('mps').eval()
        assert len(moe.experts) == 4 * g and all((not p.requires_grad for e in moe.experts for p in e.parameters()))
        assert torch.equal(moe.router.weight.detach().cpu(), parent_vectors(SEED, torch.float32).repeat_interleave(g, 0))
        accum = {'bd_diff': 0.0, 'bd_ref': 0.0, 'mb_diff': 0.0, 'mb_ref': 0.0, 'logit_sq': 0.0, 'logit_n': 0, 'max_logit': 0.0, 'kl_sum': 0.0, 'tokens': 0}

        def capture(_module, inputs):
            accum['layer_input'] = inputs[0].detach()
        hook = dense.register_forward_pre_hook(capture)
        with torch.inference_mode():
            for i in range(0, 256, 4):
                ids = data['input_ids'][i:i + 4].to('mps')
                mask = data['attention_mask'][i:i + 4].to('mps')
                base.gpt_neox.layers[3].mlp = dense
                logits_dense = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
                x = accum.pop('layer_input')
                a, b, c = three_functions(dense, moe, x)
                valid = mask.reshape(-1).bool()
                a, b, c = (a.reshape(-1, 512)[valid], b.reshape(-1, 512)[valid], c.reshape(-1, 512)[valid])
                accum['bd_diff'] += float((b - a).float().square().sum().cpu())
                accum['bd_ref'] += float(a.float().square().sum().cpu())
                accum['mb_diff'] += float((c - b).float().square().sum().cpu())
                accum['mb_ref'] += float(b.float().square().sum().cpu())
                base.gpt_neox.layers[3].mlp = moe
                logits_moe = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
                ld = logits_dense[mask.bool()]
                lm = logits_moe[mask.bool()]
                diff = lm - ld
                accum['logit_sq'] += float(diff.float().square().sum().cpu())
                accum['logit_n'] += diff.numel()
                accum['max_logit'] = max(accum['max_logit'], float(diff.abs().max().cpu()))
                p = ld.float().log_softmax(-1)
                q = lm.float().log_softmax(-1)
                kl = (p.exp() * (p - q)).sum(-1)
                accum['kl_sum'] += float(kl.float().sum().cpu())
                accum['tokens'] += int(mask.sum().cpu())
                del logits_dense, logits_moe, ld, lm, diff, p, q, kl
        hook.remove()
        base.gpt_neox.layers[3].mlp = dense
        bd = math.sqrt(accum['bd_diff'] / max(accum['bd_ref'], 1e-30))
        mb = math.sqrt(accum['mb_diff'] / max(accum['mb_ref'], 1e-30))
        mean_kl = accum['kl_sum'] / accum['tokens']
        results[arm] = {'sequences': 256, 'valid_tokens': accum['tokens'], 'relF_block_dense': bd, 'relF_moe_block': mb, 'max_abs_logit_difference': accum['max_logit'], 'rms_logit_difference': math.sqrt(accum['logit_sq'] / accum['logit_n']), 'mean_token_KL': mean_kl, 'pass': bd <= 0.0001 and mb <= 1e-05 and (mean_kl <= 1e-06)}
        print(arm, results[arm], flush=True)
        del moe
        torch.mps.empty_cache()
    return results

def main():
    data = torch.load(ROOT / 'work/s1_calibration_gate_256.pt', map_location='cpu', weights_only=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'cpu':
        result = json.loads((OUT / 'NUMERICAL_GATE.json').read_text())
        result['cpu_float64'] = cpu_gate(data)
    else:
        result = {'cpu_float64': cpu_gate(data), 'mps_float32': mps_gate(data)}
    result['pass'] = all((v['pass'] for section in ('cpu_float64', 'mps_float32') for v in result[section].values()))
    (OUT / 'NUMERICAL_GATE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('NUMERICAL PASS', result['pass'], flush=True)
if __name__ == '__main__':
    main()
