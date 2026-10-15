# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""M3: only frozen all-point primary criteria; writes decision before reports."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math, numpy as np, pyarrow.parquet as pq
from common import *

def run():
    verify_seal()
    freeze = read(PKG / 'P2L_PROTOCOL_FREEZE.json')
    completion = read(OUT / 'P2L_TRAINING_COMPLETION.json')
    rows = pq.read_table(OUT / 'P2L_TRAJECTORIES.parquet').to_pylist()
    preds = csvread(OUT / 'P2L_FROZEN_PREDICTIONS.csv')
    fits = []
    modes = []
    cells = []
    for pred in preds:
        i = int(pred['cell'])
        N = int(pred['N_cell'])
        gamma = float(pred['gamma_plus'])
        cellfits = []
        for sign in [-1, 1]:
            arm = sorted([x for x in rows if x['cell'] == i and x['sign'] == sign], key=lambda x: x['update'])
            x = np.array([x['update'] for x in arm], dtype=float)
            y = np.log([x['A1'] for x in arm])
            dx = x - x.mean()
            dy = y - y.mean()
            g = float(dx @ dy / (dx @ dx))
            b = float(y.mean() - g * x.mean())
            ss = float(np.sum((y - b - g * x) ** 2))
            total = float(dy @ dy)
            r2 = 1 - ss / total if total else 0.0
            complete = list(x) == list(range(N + 1))
            local = all((not a['locality_violation'] and a['F1'] >= 0.95 and (a['C_plus'] >= 0.95) and (a['sign_pair_antisymmetry'] <= 0.05) and (0.7 <= a['full_amplitude_ratio'] <= 1.4) for a in arm))
            fit = dict(cell=i, r_eta=float(pred['r_eta']), sign=sign, N_cell=N, recorded_points=len(arm), gamma_hat=g, intercept=b, R2=r2, predicted_gamma=gamma, full_horizon=complete, locality_valid=local, rate_R2_pass=r2 >= freeze['rate_r2_min'])
            fits.append(fit)
            cellfits.append(fit)
            modes.append(dict(cell=i, r_eta=float(pred['r_eta']), sign=sign, F1_min=min((a['F1'] for a in arm)), C_plus_min=min((a['C_plus'] for a in arm)), E_pm_max=max((a['sign_pair_antisymmetry'] for a in arm)), amplitude_ratio_min=min((a['full_amplitude_ratio'] for a in arm)), amplitude_ratio_max=max((a['full_amplitude_ratio'] for a in arm)), complete_horizon=complete, mode_preservation_pass=complete and all((a['F1'] >= 0.95 and a['C_plus'] >= 0.95 for a in arm)), locality_pass=complete and local))
        mean = float(np.mean([a['gamma_hat'] for a in cellfits]))
        asym = abs(cellfits[0]['gamma_hat'] - cellfits[1]['gamma_hat']) / max(abs(mean), 1e-08)
        valid = all((a['full_horizon'] and a['locality_valid'] and a['rate_R2_pass'] for a in cellfits)) and asym <= freeze['sign_rate_asymmetry_max']
        cells.append(dict(cell=i, r_eta=float(pred['r_eta']), gamma_mean=mean, predicted_gamma=gamma, sign_rate_asymmetry=asym, sign_asymmetry_pass=asym <= 0.1, RATE_VALID=valid))
    baseline = next((a for a in cells if a['r_eta'] == 1))
    ratios = []
    for cell in cells:
        if cell['r_eta'] == 1:
            continue
        obs = cell['gamma_mean'] / baseline['gamma_mean']
        pred = cell['predicted_gamma'] / baseline['predicted_gamma']
        err = abs(obs / pred - 1)
        ratios.append(dict(r_eta=cell['r_eta'], observed_ratio=obs, predicted_ratio=pred, relative_error=err, passed=err <= freeze['mobility_ratio_relative_error_max']))
    obs = np.array([a['gamma_mean'] for a in cells])
    pred = np.array([a['predicted_gamma'] for a in cells])
    error = float(np.linalg.norm(obs - pred) / np.linalg.norm(pred))
    pearson = float(np.corrcoef(obs, pred)[0, 1])
    positive = bool(np.all(obs > 0))
    ordering = bool(np.array_equal(np.argsort(obs), np.argsort(pred)))
    direct = error <= freeze['direct_rate_relative_l2_max'] and pearson >= freeze['direct_rate_pearson_min'] and positive
    gates = dict(all_cells_rate_valid=all((a['RATE_VALID'] for a in cells)), mobility_ordering=ordering, mobility_ratios=all((a['passed'] for a in ratios)), direct_rate_replication=direct, mode_preservation=all((a['mode_preservation_pass'] for a in modes)))
    unresolved = not all((a['locality_pass'] for a in modes))
    verdict = 'P2L_MOBILITY_CONFIRMATION_UNRESOLVED' if unresolved else 'P2L_MOBILITY_CONFIRMATION_SUPPORTED' if all(gates.values()) else 'P2L_MOBILITY_CONFIRMATION_NOT_SUPPORTED'
    csvout('P2L_RATE_FITS.csv', fits)
    csvout('P2L_MOBILITY_RATIOS.csv', ratios)
    csvout('P2L_MODE_PRESERVATION.csv', modes)
    write('P2L_PRIMARY_DECISION.json', dict(verdict=verdict, decided_utc=utc(), execution_node='P2M', protocol_id=freeze['protocol_id'], gates=gates, cells=cells, direct_rate_relative_L2=error, direct_rate_Pearson=pearson, all_mean_rates_positive=positive, all_six_runs_local=not unresolved, historical_P2J_verdict='DETERMINISTIC_FIXED_BANK_R1_BRIDGE_UNRESOLVED', counters=COUNTERS, primary_sha256={n: sha(OUT / n) for n in ['P2L_PREFLIGHT_SEAL.json', 'P2L_TRAJECTORIES.parquet', 'P2L_CHECKPOINT_INDEX.json', 'P2L_TRAINING_COMPLETION.json', 'P2L_RATE_FITS.csv', 'P2L_MOBILITY_RATIOS.csv', 'P2L_MODE_PRESERVATION.csv']}))
    report = '# P2L / P2M 独立 mobility 确认：主分析\n\n结论：`' + verdict + '`。\n\n'
    report += '冻结条件：ε=0.24、ρ₀=10⁻⁴、α=0.20；完整 F0、固定序列顺序；六组符号实验。每组使用全部整数点 0..N 做 OLS，包含终点，未修改窗口或门槛。\n\n'
    report += '| r_eta | N | 理论 γ | 实测平均 γ | 符号率不对称 | RATE_VALID |\n|---:|---:|---:|---:|---:|:---:|\n'
    for a, p in zip(cells, preds):
        report += f"| {a['r_eta']:g} | {p['N_cell']} | {a['predicted_gamma']:.12g} | {a['gamma_mean']:.12g} | {a['sign_rate_asymmetry']:.5g} | {a['RATE_VALID']} |\n"
    report += f'\n三点直接率复制：相对 L2={error:.12g}（≤0.10），Pearson={pearson:.12g}（≥0.995）。\n\n'
    report += '| r_eta / 1 | 理论率比 | 实测率比 | 相对误差 |\n|---:|---:|---:|---:|\n'
    for a in ratios:
        report += f"| {a['r_eta']:g} | {a['predicted_ratio']:.12g} | {a['observed_ratio']:.12g} | {a['relative_error']:.6g} |\n"
    report += f"\n全部六组最小 R²={min((a['R2'] for a in fits)):.12g}；最小 F1={min((a['F1_min'] for a in modes)):.12g}；最小 C+={min((a['C_plus_min'] for a in modes)):.12g}；最大 E±={max((a['E_pm_max'] for a in modes)):.12g}；最大完整振幅比={max((a['amplitude_ratio_max'] for a in modes)):.12g}（≤1.40）。\n\n"
    report += '历史 P2J/P2J.1 结论保持 `DETERMINISTIC_FIXED_BANK_R1_BRIDGE_UNRESOLVED`。本次结论只涉及这一 SmolLM2 layer-20 构造的确定性、固定 F0 mobility；不外推到随机 minibatch SGD、其他层或模型。\n'
    if verdict.endswith('_SUPPORTED') and (not verdict.endswith('_NOT_SUPPORTED')):
        report += '\n项目层面：此前未解决的确定性 mobility 预测已获得独立确认支持。\n'
    textout('P2L_PRIMARY_REPORT.md', report)
    print(verdict + '; direct relative L2=' + str(error) + '; Pearson=' + str(pearson), flush=True)
if __name__ == '__main__':
    run()
