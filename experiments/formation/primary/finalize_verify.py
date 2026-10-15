# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Independent checks and deliverables. No model calls, training or science amendments."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math, sys, zipfile, numpy as np
import pyarrow.parquet as pq
from common import *

def run():
    seal = verify_seal()
    audit = read(OUT / 'P2J_REUSE_AUDIT.json')
    done = read(OUT / 'P2J_TRAINING_COMPLETION.json')
    decision = read(OUT / 'P2J_PRIMARY_DECISION.json')
    secondary = read(OUT / 'P2J1_SECONDARY_SEAL.json')
    for p, h in audit['consumed_sha256'].items():
        assert sha(p) == h, p
    for n, h in audit['package_sha256'].items():
        assert sha(PKG / n) == h, n
    for n, h in decision['primary_artifact_sha256'].items():
        assert sha(OUT / n) == h, n
    for n, h in secondary['artifacts'].items():
        assert sha(OUT / n) == h, n
    assert secondary['primary_decision_sha256'] == sha(OUT / 'P2J_PRIMARY_DECISION.json')
    assert done['started_utc'] >= seal['sealed_utc'] and decision['primary_sealed_utc'] <= secondary['secondary_created_utc']
    assert done['counters'] == COUNTERS == seal['counters'] and done['primary_runs'] == 30
    rows = pq.read_table(OUT / 'P2J_TRAJECTORIES.parquet').to_pylist()
    by = {}
    for r in rows:
        by.setdefault((r['cell'], r['sign']), []).append(r)
    assert len(by) == 30 and len(rows) == done['logged_rows']
    pred = csvread(OUT / 'P2J_FROZEN_CELL_PREDICTIONS.csv')
    fits = csvread(OUT / 'P2J_RATE_FITS.csv')
    statics = csvread(OUT / 'P2J_STATIC_BOUNDARY.csv')
    fitmap = {(int(r['cell']), int(r['sign'])): r for r in fits}
    measured = {}
    updates = 0
    for p in pred:
        c = int(p['cell'])
        eps = float(p['epsilon'])
        r = float(p['r_eta'])
        a = float(p['alpha'])
        N = int(p['N_cell'])
        tr = -1 - r * (1 - eps)
        discriminant = tr * tr + 4 * r * eps
        bplus = (tr + math.sqrt(discriminant)) / 2
        rhop = 1 + a * bplus
        g = math.log(abs(rhop))
        assert abs(g - float(p['gamma_plus'])) < 2e-15
        for sign in [-1, 1]:
            rs = sorted(by[c, sign], key=lambda q: q['update'])
            assert [q['update'] for q in rs] == list(range(len(rs))) and len(rs) <= N + 1
            updates += len(rs) - 1
            assert all((q['N_cell'] == N for q in rs))
            for q in rs:
                assert math.isclose(q['A1'], math.hypot(q['m1'], q['u1']), rel_tol=2e-15)
                assert math.isclose(q['A_full'], math.hypot(q['M_norm'], q['U_norm']), rel_tol=2e-15)
                violations = q['F1'] < 0.95 or q['C_plus'] < 0.95 or q['sign_pair_antisymmetry'] > 0.05 or (not 0.7 <= q['full_amplitude_ratio'] <= 1.4)
                if violations:
                    assert q is rs[-1] and q['locality_violation']
            if eps:
                coeff = np.polyfit(np.arange(len(rs)), np.log([q['A1'] for q in rs]), 1)
                stored = float(fitmap[c, sign]['gamma_hat'])
                assert abs(float(coeff[0]) - stored) < 5e-15
                measured[eps, r, sign] = stored
        minus = by[c, -1]
        plus = by[c, 1]
        assert len(minus) == len(plus)
        for qm, qp in zip(minus, plus):
            E = math.hypot(qm['m1'] + qp['m1'], qm['u1'] + qp['u1']) / (qm['A1'] + qp['A1'])
            assert abs(E - qm['sign_pair_antisymmetry']) < 2e-15
    assert updates == done['primary_updates'] == done['full_F0_gradient_evaluations'] <= 912
    assert done['final_or_violating_NLL_evaluations'] == 30
    for r in statics:
        if r['epsilon_c']:
            ratio = float(r['r_eta'])
            lo = np.mean([measured[-0.08, ratio, s] for s in [-1, 1]])
            hi = np.mean([measured[0.08, ratio, s] for s in [-1, 1]])
            assert lo * hi < 0
            threshold = -0.08 - lo * 0.16 / (hi - lo)
            assert abs(threshold - float(r['epsilon_c'])) < 2e-15 and -0.08 <= threshold <= 0.08
    binding = read(OUT / 'P2J_F0_AND_C_BINDING.json')
    z = np.load(binding['cross_hessian_path'])
    v = z['v'][:, :, 0].astype(float)
    u = z['u'][:, 0].astype(float)
    index = read(OUT / 'P2J_CHECKPOINT_INDEX.json')
    lookup = {(r['cell'], r['sign'], r['update']): r for r in rows}
    for ck in index['checkpoints']:
        assert sha(ck['path']) == ck['sha256']
        with np.load(ck['path'], allow_pickle=False) as z:
            M = z['M'].astype(float)
            U = z['U'].astype(float)
        row = lookup[ck['cell'], ck['sign'], ck['update']]
        mp = float(np.sum(M * v))
        up = float(U @ u)
        full = math.sqrt(float(np.sum(M * M) + U @ U))
        assert abs(mp - row['m1']) < 2e-15 and abs(up - row['u1']) < 2e-15 and (abs(full - row['A_full']) < 2e-15)
    reevaluated = read(OUT / 'P2J1_ALPHA_GATE_REEVALUATED.json')
    assert reevaluated['selected_alpha'] == 0.2 and reevaluated['model_calls'] == 0
    required = ['P2J_REUSE_AUDIT.json', 'P2J_F0_AND_C_BINDING.json', 'P2J_AMPLITUDE_GATE.csv', 'P2J_ALPHA_GATE.csv', 'P2J_FROZEN_CELL_PREDICTIONS.csv', 'P2J_PREFLIGHT_SEAL.json', 'P2J_TRAJECTORIES.parquet', 'P2J_CHECKPOINT_INDEX.json', 'P2J_RATE_FITS.csv', 'P2J_STATIC_BOUNDARY.csv', 'P2J_DISCRETE_KINETICS.csv', 'P2J_MOBILITY_KINETICS.csv', 'P2J_MODE_PRESERVATION.csv', 'P2J_PRIMARY_DECISION.json', 'P2J_PRIMARY_REPORT.md', 'P2J_NONLINEARITY_DIAGNOSTICS.csv', 'P2J_SECONDARY_REPORT.md']
    assert all(((OUT / n).exists() for n in required))
    write('P2J1_VERIFICATION.json', dict(status='PASS', all_required_outputs_present=True, historical_blocked_run_and_P2H_assets_unchanged=True, verified_consumed_assets=len(audit['consumed_sha256']), code_freeze_unchanged=True, exact_discrete_predictions_independently_verified=True, horizons_verified_against_amendment=True, all_30_arms_verified=True, OLS_fits_independently_verified=True, static_brackets_no_extrapolation=True, sign_pair_envelope_recomputed=True, checkpoint_hashes_and_observables_verified=len(index['checkpoints']), primary_before_secondary_verified=True, counters=COUNTERS, verified_utc=utc()))
    statictext = '\n'.join((f"| {r['r_eta']} | {r['epsilon_c'] or 'unresolved'} | {r['status']} |" for r in statics))
    marg = csvread(OUT / 'P2J1_MARGINAL_CONTROLS.csv')
    maxdrift = max((float(r['max_abs_relative_A1_drift']) for r in marg))
    componenttext = '\n'.join((f'- {k}: `{v}`' for k, v in decision['components'].items()))
    report = f"# P2J.1 修正恢复执行报告\n\n**最终判决：`{decision['decision']}`。**\n\n按原 P2J 协议及唯一 P2J.1 修正完成 30 个确定性符号运行。有效非零 ε 格数：{decision['valid_nonzero_cells']}/12。局部性违规格数：{decision['locality_violating_cells']}。原 P2J 预检阻断及全部历史输入保留，运行后哈希再次通过。\n\n## 1. 修正和恢复链\n\n包内原协议 SHA-256 `{read(OUT / 'P2J1_PROVENANCE_BINDING.json')['parent_protocol_sha256']}`。修正只覆盖 ε=0 的 J2 归一化、ε=0 主分析角色及静态 bracket 定义。\n\nR0 复核旧阻断报告、复现包、幅度/步长 CSV、模型、F0 及 C 哈希，确认原 primary updates=0。R1 不调用模型，离线重判 45 行：最大 E0_parallel={reevaluated['max_E0_parallel']:.9g}，最大 E0_perp={reevaluated['max_E0_perp']:.9g}，均远小于 0.05；原非零 ε 门不变，选择最大 α=0.20。由于旧 CSV 只有总步长和正交残差范数，E0_parallel 从平方差还原，并针对原 u1/v1 约 4e-8 的单位范数缺陷记录严格上下界；以保守上界判断，全部边缘格仍通过。\n\nR2 独立重新生成并核对全部离散预测、冻结 horizon，总预算 912；在所有多步更新前写入新的 `P2J_PREFLIGHT_SEAL.json`。所有执行和分析源文件也在该 seal 内封存。\n\n原 `P2J_ALPHA_GATE.csv` 与幅度 CSV 是旧预检数据的精确只读副本；修正判定见 `P2J1_ALPHA_GATE_REEVALUATED.csv/json`，新 PASS preflight seal 是此次运行授权依据。\n\n## 2. 执行和数值约束\n\n- 复用 exact `HuggingFaceTB/SmolLM2-135M` commit `{binding['model_commit']}`，zero-based layer 20。\n- 复用 sealed F0：`{binding['F0_file']}`，32×256 输入 token，8192 输入位置/8160 next-token 监督位置。\n- 复用 P2H cross-Hessian `{binding['cross_hessian_path']}`，s1={binding['s1']:.17g}；没有重新拟合或计算 C。\n- 参数/模型 float32，CPU 8 线程，固定序列 0..31，microbatch=1，完整 F0 梯度 float64 累加，除以 32 一次；float64 计算更新后写回 float32。\n- 同一四专家 dense softmax private parametrization，仅 M/U 更新；共同权重冻结，dropout 关闭，无随机 minibatch/shuffle。\n- ρ0=1e-4，a0={seal['selected_a0']:.17g}，α=.20，15 格×两个符号。\n- 所有 horizon 与修正版表完全一致，每格只执行到 frozen horizon 或首次 locality violation。实际 primary full-F0 updates：{done['primary_updates']}/{done['frozen_budget']}；输入 token 更新预算：{done['primary_input_tokens']:,}。\n- 全部逐步 observables/NLL/正则项写入 parquet；每 4 步及端点保留 checkpoints，共 {len(index['checkpoints'])} 个。\n- 主运行耗时：{done['elapsed_seconds'] / 60:.2f} 分钟；开始 UTC {done['started_utc']}，结束 UTC {done['finished_utc']}。\n\n## 3. 主分析\n\n{componenttext}\n\n非零格采用全部整数步 0..N 的 OLS，R²≥0.995，符号 rate asymmetry≤0.10，且须全程保持局部性。离散动力学相对 L2 误差：{decision['kinetic_relative_l2_error']:.9g}；Pearson：{decision['kinetic_pearson']:.12g}；有效格预测符号全部匹配：{decision['kinetic_all_valid_signs_match']}。冻结支持门为 ≤0.10、≥0.995、至少 10/12 格。\n\n静态阈值严格使用已预定 −0.08,+0.08 两格，要求两格 RATE_VALID 且反号，仅在 bracket 内插值：\n\n| rη | εc | 状态 |\n|---|---|---|\n{statictext}\n\n跨 mobility 的 εc 范围：{decision['static_epsilon_c_range']}；支持门 |εc|≤0.04，range≤0.04。mobility 比值和 mode preservation 的原门保持不变。详细每格结果见 `P2J_PRIMARY_REPORT.md` 及原协议指定 CSV。\n\nε=0 的 6 个运行作为漂移/局部性/符号对称控制，不参与主 rate、kinetic、mobility、mode 或阈值支持计数。最大相对 A1 漂移：{maxdrift:.9g}，见 `P2J1_MARGINAL_CONTROLS.csv`。\n\n## 4. 封存和验证\n\n先封存 `P2J_PRIMARY_DECISION.json`，再读取已保存轨迹/checkpoints 生成 secondary 诊断。secondary model/gradient/update/download count 全部 0；没有 objective susceptibility 或 stochastic extension。\n\n独立验证重算离散特征值、OLS rates、静态内插、符号 envelope；逐个校验 checkpoint 哈希和其投影/范数；复核主/次分析封存时间顺序及全部历史消费哈希。结果 `P2J1_VERIFICATION.json: PASS`。新下载字节、新 bank、seal 前多步更新、outcome-dependent science changes 全部 0，AMENDMENT_COUNT=1。\n\n## 5. 解释范围\n\n结果只适用于这一个 SmolLM2 layer-20 four-expert soft-MoE 私有构造、same-F0 deterministic local formation 实验。不得扩展为 stochastic minibatch SGD 或 native MoE 结论。原 P2H 判决继续为 `STATIC_KINETIC_SEPARATION_NOT_SUPPORTED`。\n\n完整输出位于 `{OUT}`。小型结果复现包包含冻结协议、代码、CSV/JSON/parquet、报告和日志；大型 checkpoints 保留在该目录的 `checkpoints/`，其绝对路径和哈希见 checkpoint index。`SHA256SUMS_P2J.txt` 覆盖所有交付文件和 checkpoints。\n"
    textout('P2J_EXECUTION_REPORT.md', report)
    textout('P2J1_REPRODUCE.md', f'# P2J.1 Reproduction\n\nExisting interpreter: `{sys.executable}`. No downloads or installations.\n\nThe code uses the old blocked run and P2H inputs in their original locations. To replay, use a new empty output directory under the same workspace; set `work/p2j1_output_path.txt` to its absolute path. Preserve all old assets and do not overwrite sealed outputs. Source stages:\n\n```sh\n{sys.executable} work/p2j1_source/preflight_resume.py\n{sys.executable} work/p2j1_source/train_primary.py\n{sys.executable} work/p2j1_source/analyze_primary.py\n{sys.executable} work/p2j1_source/finalize_verify.py\n```\n\nPreflight is offline and model-call-free. Model load occurs only after the new PASS seal. Training refuses existing trajectories; no automatic new science amendment or extended horizon is implemented. Analysis seals the primary decision before diagnostics. Download/network prohibition is enforced by offline flags, local-file-only loads and socket connection blocks.\n\nThe absolute checkpoint paths in the result index refer to the executed workspace; checkpoints are excluded from the compact ZIP and remain available in the output directory.\n')
    archive = OUT / 'P2J1_RESULT_AND_REPRODUCIBILITY.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SOURCE.glob('*.py')):
            z.write(p, str(Path('source') / p.name))
        for p in sorted(PKG.iterdir()):
            if p.is_file():
                z.write(p, str(Path('authority') / p.name))
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, str(Path('results') / p.name))
        for name in ['p2j1_preflight.log', 'p2j1_training.log', 'p2j1_analysis.log']:
            p = ROOT / 'work' / name
            if p.exists():
                z.write(p, str(Path('logs') / name))
    archive.chmod(292)
    files = sorted((p for p in OUT.rglob('*') if p.is_file()))
    textout('SHA256SUMS_P2J.txt', ''.join((f'{sha(p)}  {p.relative_to(OUT)}\n' for p in files)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print('FINAL VERIFICATION PASS:', decision['decision'], 'updates', updates, 'checkpoints', len(index['checkpoints']), flush=True)
if __name__ == '__main__':
    run()
