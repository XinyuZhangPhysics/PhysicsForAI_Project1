# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Recompute frozen selected-sequence routing diagnostics without changing outcomes."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import json
from pathlib import Path
import numpy as np
import torch
from transformers import OlmoeForCausalLM
BASE = Path(__file__).parent
OUT = BASE / 'u4_run'
MODEL = _p1_path('path_f2699b70cfbf')
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']
GROUPS = [(s, d) for s in ('validation', 'test') for d in DOMAINS]
selected = json.loads((BASE / 'u4_selected_sequences.json').read_text())
torch.set_num_threads(8)
model = OlmoeForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager')
model.eval()
stats = {}
for group in GROUPS:
    freq = np.zeros(64, dtype=np.int64)
    weight_sum = np.zeros(64, dtype=np.float64)
    selected_count = np.zeros(64, dtype=np.int64)
    conditional_entropy = []
    sparse_entropy = []
    selected_mass = []
    rows = [r for r in selected if (r['split'], r['domain']) == group]
    for row in rows:
        ids = torch.tensor(row['token_ids'], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model.model(input_ids=ids, use_cache=False, output_router_logits=True)
            logits = output.router_logits[15]
            weights, labels = torch.topk(torch.softmax(logits.float(), dim=-1), 8, dim=-1)
        a = labels.numpy().reshape(-1)
        w = weights.numpy().reshape(-1)
        np.add.at(freq, a, 1)
        np.add.at(selected_count, a, 1)
        np.add.at(weight_sum, a, w)
        ws = weights.numpy()
        mass = ws.sum(axis=1)
        q = ws / mass[:, None]
        conditional_entropy.extend((-np.sum(q * np.log(q), axis=1)).tolist())
        sparse_entropy.extend((-np.sum(ws * np.log(ws), axis=1)).tolist())
        selected_mass.extend(mass.tolist())
    total_tokens = 16 * 128
    stats[group] = {'freq': freq, 'weight_sum': weight_sum, 'selected_count': selected_count, 'mean_conditional_entropy': float(np.mean(conditional_entropy)), 'mean_sparse_entropy': float(np.mean(sparse_entropy)), 'mean_selected_mass': float(np.mean(selected_mass)), 'distinct_experts': int(np.count_nonzero(freq)), 'total_tokens': total_tokens}
    print(json.dumps({'split': group[0], 'domain': group[1], 'distinct_experts': stats[group]['distinct_experts']}), flush=True)
with (OUT / 'u4_routing_diagnostics.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['split', 'domain', 'expert', 'selection_count', 'selection_frequency_per_token', 'mean_weight_when_selected', 'mean_weight_per_token', 'mean_top8_conditional_entropy', 'mean_sparse_top8_entropy', 'mean_selected_mass', 'distinct_experts', 'token_positions'])
    for group in GROUPS:
        st = stats[group]
        for j in range(64):
            n = int(st['selected_count'][j])
            w.writerow([*group, j, n, int(st['freq'][j]) / st['total_tokens'], float(st['weight_sum'][j] / n) if n else 0.0, float(st['weight_sum'][j] / st['total_tokens']), st['mean_conditional_entropy'], st['mean_sparse_entropy'], st['mean_selected_mass'], st['distinct_experts'], st['total_tokens']])
with (OUT / 'u4_routing_overlap.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['split_a', 'domain_a', 'split_b', 'domain_b', 'mean_shared_top8_count_over_8', 'mean_weighted_overlap'])
    for g1 in GROUPS:
        for g2 in GROUPS:
            a = stats[g1]['freq'] / stats[g1]['total_tokens']
            b = stats[g2]['freq'] / stats[g2]['total_tokens']
            wa = stats[g1]['weight_sum'] / stats[g1]['total_tokens']
            wb = stats[g2]['weight_sum'] / stats[g2]['total_tokens']
            w.writerow([*g1, *g2, float(a @ b / 8), float(wa @ wb)])
print(json.dumps({'status': 'PASS', 'groups': len(GROUPS), 'selected_sequences': len(selected)}), flush=True)
