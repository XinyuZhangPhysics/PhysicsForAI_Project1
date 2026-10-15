# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""M4: separate analytic eigen/horizon and least-squares/checkpoint recomputation."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math, numpy as np, pyarrow.parquet as pq, torch, zipfile, datetime
from common import *

def close(a, b, tol=2e-12):
    assert math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol), (a, b)

def run():
    seal = verify_seal()
    audit = read(OUT / 'P2L_REUSE_AUDIT.json')
    d = read(OUT / 'P2L_PRIMARY_DECISION.json')
    done = read(OUT / 'P2L_TRAINING_COMPLETION.json')
    index = read(OUT / 'P2L_CHECKPOINT_INDEX.json')
    for n, h in d['primary_sha256'].items():
        assert sha(OUT / n) == h
    assert done['source_sha256'] == seal['source_sha256']
    assert done['preflight_seal_sha256'] == index['preflight_seal_sha256'] == sha(OUT / 'P2L_PREFLIGHT_SEAL.json')
    preds = csvread(OUT / 'P2L_FROZEN_PREDICTIONS.csv')
    fits = csvread(OUT / 'P2L_RATE_FITS.csv')
    ratios = csvread(OUT / 'P2L_MOBILITY_RATIOS.csv')
    modes = csvread(OUT / 'P2L_MODE_PRESERVATION.csv')
    rows = pq.read_table(OUT / 'P2L_TRAJECTORIES.parquet').to_pylist()
    z = np.load(audit['cross_hessian'], allow_pickle=False)
    v = z['v'][:, :, 0].copy()
    u = z['u'][:, 0].copy()
    vd = v.astype(float)
    ud = u.astype(float)
    analytic = []
    armrates = {}
    extrema = []
    cache = {}
    checkpoint_checks = 0
    assert len(index['checkpoints']) == len(rows) == done['logged_rows']
    rowmap = {(a['cell'], a['sign'], a['update']): a for a in rows}
    assert len(rowmap) == len(rows)
    predmap = {int(p['cell']): p for p in preds}
    for p in preds:
        r = float(p['r_eta'])
        alpha = float(p['alpha'])
        eps = float(p['epsilon'])
        tr = -1 - r * (1 - eps)
        det = -r * eps
        branch = (tr + math.sqrt(tr * tr - 4 * det)) / 2
        rho = 1 + alpha * branch
        other = 1 + alpha * (tr - math.sqrt(tr * tr - 4 * det)) / 2
        g = math.log1p(alpha * branch)
        N = math.ceil(math.log(1.25) / g)
        while rho ** (N - 1) >= 1.25:
            N -= 1
        while rho ** N < 1.25:
            N += 1
        close(rho, p['rho_plus'])
        close(other, p['rho_minus'])
        close(g, p['gamma_plus'])
        assert N == int(p['N_cell']) and rho ** N <= 1.35 and (rho ** (N - 1) < 1.25)
        ev = np.array([1.0, 1 + branch])
        ev /= np.linalg.norm(ev)
        close(ev[0], p['e_M'])
        close(ev[1], p['e_U'])
        T = np.array([[float(p['T00']), float(p['T01'])], [float(p['T10']), float(p['T11'])]])
        assert np.linalg.norm(T @ ev - rho * ev) < 1e-14
        analytic.append(dict(r_eta=r, rho_plus=rho, gamma_plus=g, N=N, endpoint=rho ** N))
    for cp in index['checkpoints']:
        assert sha(cp['path']) == cp['sha256']
        checkpoint_checks += 1
        k = (cp['cell'], cp['sign'], cp['update'])
        a = rowmap[k]
        p = predmap[cp['cell']]
        q = np.load(cp['path'], allow_pickle=False)
        M = q['M']
        U = q['U']
        assert M.shape == v.shape and U.shape == u.shape and (M.dtype == U.dtype == np.float32)
        assert np.isfinite(M).all() and np.isfinite(U).all()
        m = float(np.sum(M.astype(float) * vd))
        nu = float(np.dot(U.astype(float), ud))
        mn = float(np.linalg.norm(M.astype(float)))
        un = float(np.linalg.norm(U.astype(float)))
        full = math.hypot(mn, un)
        A = math.hypot(m, nu)
        C = (float(p['e_M']) * m + float(p['e_U']) * nu) ** 2 / (A * A)
        for name, value in [('m1', m), ('u1', nu), ('A1', A), ('M_norm', mn), ('U_norm', un), ('A_full', full), ('F1', A * A / full ** 2), ('C_plus', C)]:
            close(value, a[name])
        cache[k] = (m, nu, full, A, C, A * A / full ** 2)
        if cp['update'] == 0:
            expectedM = cp['sign'] * float(p['a0']) * float(p['e_M']) * torch.from_numpy(v)
            expectedU = cp['sign'] * float(p['a0']) * float(p['e_U']) * torch.from_numpy(u)
            assert np.array_equal(M, expectedM.numpy()) and np.array_equal(U, expectedU.numpy())
    for a in rows:
        k = (a['cell'], a['sign'], a['update'])
        q = cache[k]
        q0 = cache[a['cell'], a['sign'], 0]
        close(q[2] / q0[2], a['full_amplitude_ratio'])
        pair = cache[a['cell'], -a['sign'], a['update']]
        E = math.hypot(q[0] + pair[0], q[1] + pair[1]) / (q[3] + pair[3])
        close(E, a['sign_pair_antisymmetry'])
        reasons = []
        if q[5] < 0.95:
            reasons.append('LEADING_PAIR_CAPTURE')
        if q[4] < 0.95:
            reasons.append('EIGENVECTOR_ALIGNMENT')
        if E > 0.05:
            reasons.append('SIGN_ANTISYMMETRY')
        if not 0.7 <= q[2] / q0[2] <= 1.4:
            reasons.append('FULL_AMPLITUDE_ENVELOPE')
        assert a['locality_reasons'] == ';'.join(reasons)
    for f in fits:
        i = int(f['cell'])
        sign = int(f['sign'])
        arm = sorted([a for a in rows if a['cell'] == i and a['sign'] == sign], key=lambda a: a['update'])
        x = np.array([a['update'] for a in arm], float)
        y = np.log([cache[i, sign, a['update']][3] for a in arm])
        design = np.column_stack([np.ones(len(x)), x])
        b, g = np.linalg.lstsq(design, y, rcond=None)[0]
        r2 = 1 - float(np.sum((y - design @ np.array([b, g])) ** 2) / np.sum((y - y.mean()) ** 2))
        close(g, f['gamma_hat'])
        close(b, f['intercept'])
        close(r2, f['R2'])
        armrates[i, sign] = float(g)
        me = next((m for m in modes if int(m['cell']) == i and int(m['sign']) == sign))
        ex = dict(cell=i, sign=sign, F1_min=min((a['F1'] for a in arm)), C_plus_min=min((a['C_plus'] for a in arm)), E_pm_max=max((a['sign_pair_antisymmetry'] for a in arm)), amplitude_ratio_min=min((a['full_amplitude_ratio'] for a in arm)), amplitude_ratio_max=max((a['full_amplitude_ratio'] for a in arm)))
        for name in ['F1_min', 'C_plus_min', 'E_pm_max', 'amplitude_ratio_min', 'amplitude_ratio_max']:
            close(ex[name], me[name])
        assert list(x) == list(range(int(f['N_cell']) + 1)) or arm[-1]['locality_violation']
        extrema.append(ex)
    observed = np.array([(armrates[i, -1] + armrates[i, 1]) / 2 for i in range(3)])
    predicted = np.array([a['gamma_plus'] for a in analytic])
    asym = []
    for i, cell in enumerate(d['cells']):
        close(observed[i], cell['gamma_mean'])
        sign_asym = abs(armrates[i, 1] - armrates[i, -1]) / max(abs(observed[i]), 1e-08)
        close(sign_asym, cell['sign_rate_asymmetry'])
        asym.append(sign_asym)
    ratio_checks = []
    for a in ratios:
        i = 0 if float(a['r_eta']) == 0.25 else 2
        obs = observed[i] / observed[1]
        pred = predicted[i] / predicted[1]
        error = abs(obs / pred - 1)
        close(obs, a['observed_ratio'])
        close(pred, a['predicted_ratio'])
        close(error, a['relative_error'])
        assert (error <= 0.1) == (a['passed'] == 'True')
        ratio_checks.append(error)
    direct = float(math.sqrt(sum((observed - predicted) ** 2) / sum(predicted ** 2)))
    dc = observed - observed.mean()
    pc = predicted - predicted.mean()
    pearson = float(np.dot(dc, pc) / math.sqrt(np.dot(dc, dc) * np.dot(pc, pc)))
    close(direct, d['direct_rate_relative_L2'])
    close(pearson, d['direct_rate_Pearson'])
    all_local = all((not a['locality_violation'] for a in rows)) and done['primary_updates'] == 78
    rate_valid = all((float(f['R2']) >= 0.995 for f in fits)) and all((a <= 0.1 for a in asym)) and all_local
    mode = all((a['F1_min'] >= 0.95 and a['C_plus_min'] >= 0.95 for a in extrema)) and all_local
    ordering = list(np.argsort(observed)) == list(np.argsort(predicted))
    replication = direct <= 0.1 and pearson >= 0.995 and all(observed > 0)
    gates = dict(all_cells_rate_valid=rate_valid, mobility_ordering=ordering, mobility_ratios=all((a <= 0.1 for a in ratio_checks)), direct_rate_replication=replication, mode_preservation=mode)
    assert gates == d['gates']
    verdict = 'P2L_MOBILITY_CONFIRMATION_UNRESOLVED' if not all_local else 'P2L_MOBILITY_CONFIRMATION_SUPPORTED' if all(gates.values()) else 'P2L_MOBILITY_CONFIRMATION_NOT_SUPPORTED'
    assert verdict == d['verdict']
    assert d['historical_P2J_verdict'] == 'DETERMINISTIC_FIXED_BANK_R1_BRIDGE_UNRESOLVED'
    assert done['primary_updates'] == done['full_F0_gradient_evaluations'] <= 78 and done['primary_input_tokens'] == done['primary_updates'] * 8192
    assert done['primary_runs'] == 6 and len(done['cell_status']) == 3
    for cs, p in zip(done['cell_status'], preds):
        assert cs['frozen_horizon'] == int(p['N_cell'])
        if cs['status'] == 'HORIZON_COMPLETE':
            assert cs['completed_update'] == cs['frozen_horizon']
        else:
            assert cs['violation'] is not None
    parse = datetime.datetime.fromisoformat
    assert parse(seal['sealed_utc']) < parse(done['started_utc']) < parse(done['finished_utc']) < parse(d['decided_utc'])
    for p, h in audit['consumed_sha256'].items():
        assert sha(p) == h, p
    for k, v in COUNTERS.items():
        assert v == d['counters'][k] == done['counters'][k] == 0
    verification = dict(status='PASS', verified_utc=utc(), verdict=verdict, analytic_predictions=analytic, independent_OLS_method='numpy.linalg.lstsq from checkpoint projections; training analysis used centered covariance', independent_eigen_method='closed-form eigenvalues of dimensionless generator plus first-row eigenvector', checkpoint_hashes_verified=checkpoint_checks, all_integer_checkpoint_observables_verified=True, locality_extrema=extrema, sign_rate_asymmetries=asym, mobility_ratio_relative_errors=ratio_checks, direct_rate_relative_L2=direct, direct_rate_Pearson=pearson, provenance_order_verified=True, seal_utc=seal['sealed_utc'], training_started_utc=done['started_utc'], training_finished_utc=done['finished_utc'], decision_utc=d['decided_utc'], exact_reused_assets_unchanged=True, primary_updates=done['primary_updates'], primary_runs=6, counters=COUNTERS, training_rerun_updates=0)
    write('P2L_VERIFICATION.json', verification)
    report = '# P2M 执行报告\n\n结论：`' + verdict + '`；独立复核：`PASS`。\n\n'
    report += f"完成 {done['primary_runs']} 组确定性实验，完整 F0 更新 {done['primary_updates']}/78 次，记录 {len(rows)} 个整数步骤观测，保存并核验 {checkpoint_checks} 份逐步 checkpoint。冻结 horizon 分别为 24、9、6，两种符号均包含终点；无 rounding、终点豁免或事后门槛调整。\n\n"
    report += 'M0：指定 commit 的本地模型、32×256 F0（含逐序列 token hash）、谱 s1/u1/v1、ρ₀=10⁻⁴ 和 P2J.1 训练前 α=0.20 绑定均通过核验。原 P2J alpha FAIL 文件未改写。旧历史目录仅只读访问。\n\n'
    report += '预封存实现检查中发现 Python 保留字误用和 MODEL_ASSET.json 元数据旁证未列入旧 consumed 表的断言问题，已在任何训练及 outcome 访问前修正。必需模型/数据/谱资产未出现 hash 不符；元数据内容亦与原封存模型文件 hash 逐一一致。检查记录保留于复现包。科学协议没有改动。\n\n'
    report += 'M1：只从理论重新求 T、e+、ρ+、log(ρ+) 和最小整数 horizon。预期终点最大 1.302385023255163，小于预注册 1.35。资产和执行/分析/验证源码一并封存，封存前训练更新数及打开多步 outcome 数均为零。\n\n'
    report += 'M2：CPU float32 参数、float64 梯度累加；固定 32 序列顺序，每序列 255 个监督位置，共 8160 个监督 token；32 个序列均值梯度之和仅除以 32 一次，并加入固定二次正则。被冻结的前 20 层仅缓存其确定性 hidden states，suffix 每步计算当前真实 loss 的完整 F0 梯度；仅 M/U 更新。复用旧运行环境与既有实现，不下载依赖。\n\n'
    report += 'M3：先写主决定，再写报告，未执行额外描述性/静态阈值/随机训练/objective 分析。详见 P2L_PRIMARY_REPORT.md。\n\n'
    report += 'M4：以闭式特征值和通用 least-squares 独立复算预测、horizon、OLS、率比和主判断；从全部逐整数 checkpoint 复算 locality/mode/sign 指标，核验所有 checkpoint hash、输入 hash、源码 hash 及 seal→training→decision 顺序。未增加训练预算。\n\n'
    report += '计数器：NEW_MODEL_DOWNLOAD_BYTES=0；NEW_DATA_DOWNLOAD_BYTES=0；NEW_BANK_COUNT=0；OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT=0。网络调用在训练入口被禁用，from_pretrained 仅允许已验证本地文件。\n\n'
    report += '当前新聊天和独立 P2M 目录满足会话隔离要求；未建立额外用户聊天。包内 Astra/High 为推荐模型配置，执行未以模型选择建议改动科学协议，也未声称已切换当前聊天模型。\n\n'
    report += '历史 P2J/P2J.1 保持 `DETERMINISTIC_FIXED_BANK_R1_BRIDGE_UNRESOLVED`。'
    if verdict == 'P2L_MOBILITY_CONFIRMATION_SUPPORTED':
        report += ' 项目层面，此前未解决的确定性 mobility 预测获得独立确认支持。'
    report += ' 结论限于指定 SmolLM2 layer-20 soft-MoE 的固定 F0 确定性构造，不涉及随机 SGD 或泛化能力。\n'
    textout('P2L_EXECUTION_REPORT.md', report)
    zp = OUT / 'P2L_REPRODUCIBILITY_SOURCE.zip'
    assert not zp.exists()
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in SOURCE.glob('*.py'):
            archive.write(p, 'source/' + p.name)
        for p in PKG.iterdir():
            if p.is_file():
                archive.write(p, 'frozen_package/' + p.name)
        for n in ['p2m_training.log', 'hf_cache_inventory.json', 'P2M_PREFLIGHT_IMPLEMENTATION_CHECK.json']:
            p = ROOT / 'work' / n
            if p.exists():
                archive.write(p, 'runtime/' + n)
    zp.chmod(292)
    manifest = ''.join((sha(p) + '  ' + str(p.relative_to(OUT)) + '\n' for p in sorted(OUT.rglob('*')) if p.is_file()))
    textout('SHA256SUMS_P2L.txt', manifest)
    for line in (OUT / 'SHA256SUMS_P2L.txt').read_text().splitlines():
        h, n = line.split(maxsplit=1)
        assert sha(OUT / n) == h, n
    print('M4 PASS: ' + str(checkpoint_checks) + ' checkpoints; all hashes, predictions, fits, ratios, extrema, decisions and provenance order verified', flush=True)
if __name__ == '__main__':
    run()
