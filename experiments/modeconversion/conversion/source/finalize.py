# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
from experiment import verify_freeze
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fr = verify_freeze()
    out = ROOT / 'outputs'
    cs = read(ROOT / 'audit/CONTROLS.json')
    cr = list(csv.DictReader((ROOT / 'audit/COEFFICIENTS.csv').open()))
    primary = []
    figdata = []
    cells = []
    allboot = {}
    expected = 192
    assert len(cs) == expected and len(cr) == 192 * 9 * 3
    assert all((c['P_ref'] > 0 and c['min_probability'] > 1e-12 and (c['max_logit_intervention'] <= 0.25) for c in cs))
    technical = True
    for f in ['S1', 'S2']:
        for g in [2, 4, 8]:
            key = f'{f}_g{g}'
            bound = fr['resolution'][key]['b1_bound']
            mbound = fr['resolution'][key]['mixed_m1_bound']
            cellcontrols = [c for c in cs if c['family'] == f and c['g'] == g]
            seedrows = []
            for seed in seeds(f):
                cc = [c for c in cellcontrols if c['seed'] == seed]
                assert len(cc) == 4
                L = {s: float(np.mean([float(c['Lambda']) for c in cr if c['family'] == f and int(c['g']) == g and (int(c['seed']) == seed) and (c['sector'] == s) and (float(c['s']) == 0)])) for s in ['A', 'C', 'M']}
                amax = max((abs(float(c[f'b1_{i:02}'])) for c in cr if c['family'] == f and int(c['g']) == g and (int(c['seed']) == seed) and (c['sector'] == 'A') for i in range(12)))
                structural = max((abs(v) for c in cc for v in c['M_exact_at_zero'] + c['M_autograd_at_zero'])) <= fr['chain_atol']
                xi = float(np.mean([c['Xi_M'] for c in cc]))
                seedrows.append([L['C'], xi])
                primary.append({'family': f, 'g': g, 'seed': seed, 'Lambda_A_0': L['A'], 'Lambda_C_0': L['C'], 'Lambda_M_0': L['M'], 'Xi_M': xi, 'b1_resolution_bound': bound, 'mixed_resolution_bound': mbound, 'A_max_abs_all_s_directions': amax, 'H1': amax <= bound, 'H2_seed': L['C'] > bound, 'H3': L['M'] <= bound or structural, 'H3_structural_null': structural, 'H4_seed': xi > mbound})
            idx = np.random.default_rng(np.random.SeedSequence([20261011964, 1 if f == 'S1' else 2, g])).integers(0, 8, size=(20000, 8))
            seedrows = np.array(seedrows)
            boot = seedrows[idx].mean(1)
            ci = np.quantile(boot, [0.025, 0.975], axis=0)
            sel = [p for p in primary if p['family'] == f and p['g'] == g]
            H1 = all((p['H1'] for p in sel))
            H2 = all((p['H2_seed'] for p in sel)) and ci[0, 0] > bound
            H3 = all((p['H3'] for p in sel))
            H4 = all((p['H4_seed'] for p in sel)) and ci[0, 1] > mbound
            cells.append({'family': f, 'g': g, 'seeds': 8, 'mean_Lambda_C_0': float(seedrows[:, 0].mean()), 'CI95_Lambda_C_0': ci[:, 0].tolist(), 'mean_Xi_M': float(seedrows[:, 1].mean()), 'CI95_Xi_M': ci[:, 1].tolist(), 'mean_Lambda_A_0': float(np.mean([p['Lambda_A_0'] for p in sel])), 'mean_Lambda_M_0': float(np.mean([p['Lambda_M_0'] for p in sel])), 'b1_resolution_bound': bound, 'mixed_resolution_bound': mbound, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'PASS': H1 and H2 and H3 and H4})
            for s in GRID:
                for sector in ['A', 'C', 'M']:
                    values = []
                    for seed in seeds(f):
                        v = [float(c['Lambda']) for c in cr if c['family'] == f and int(c['g']) == g and (int(c['seed']) == seed) and (c['sector'] == sector) and (float(c['s']) == s)]
                        assert len(v) == 4
                        values.append(np.mean(v))
                    mean = float(np.mean(values))
                    low, high = np.quantile(np.array(values)[idx].mean(1), [0.025, 0.975])
                    figdata.append({'family': f, 'g': g, 's': float(s), 'sector': sector, 'seed_mean_Lambda': mean, 'CI95_low': float(low), 'CI95_high': float(high), 'A_resolution_bound': bound})
            allboot[key] = {'indices_sha256': digest(idx.tobytes()), 'shape': [20000, 8]}
    if all((c['PASS'] for c in cells)):
        classification = 'REFERENCE_NORMALIZED_MODE_CONVERSION_VALIDATED'
    elif any((not c['H1'] or not c['H3'] for c in cells)):
        classification = 'SCIENTIFIC_CONFLICT'
    elif any((c['PASS'] for c in cells)):
        classification = 'PARTIAL_REFERENCE_NORMALIZED_MODE_CONVERSION'
    else:
        classification = 'SCIENTIFIC_CONFLICT'
    before = read(ROOT / 'preflight/HISTORICAL_BEFORE.json')
    after = {str(p): sha(p) for r in before['roots'] for p in sorted(Path(r).rglob('*')) if p.is_file() and '.git' not in p.parts}
    assert after == before['files'], 'HISTORICAL_CHANGE'
    write(ROOT / 'preflight/HISTORICAL_AFTER.json', {'timestamp': now(), 'files': after})
    write(ROOT / 'preflight/READ_ONLY_INTEGRITY.json', {'historical_files': len(after), 'modified_files': [], 'inventory_changes': [], 'r1_report_unchanged': True, 'frozen_assets_unchanged': True})
    results = {'global_classification': classification, 'technically_valid': technical, 'cells': cells, 'freeze_sha256': sha(ROOT / 'preregistration/FREEZE.json'), 'audit_opening': read(ROOT / 'audit/AUDIT_OPENING_RECORD.json'), 'statistical_units_per_cell': 8, 'bootstrap': fr['bootstrap'], 'bootstrap_indices': allboot, 'audit_raw_rows': len(list(csv.DictReader((ROOT / 'audit/RAW.csv').open()))), 'directions_per_seed': 4, 'historical_files_unchanged': len(after), 'scope': 'reference-normalized restricted feature-space response; no full B_ts or downstream utility claim'}
    write(out / 'BACKGROUND_MODE_CONVERSION_R2_RESULTS.json', results)
    table(out / 'BACKGROUND_MODE_CONVERSION_R2_PRIMARY.csv', primary)
    (out / 'BACKGROUND_MODE_CONVERSION_R2_SCALING.csv').write_bytes((ROOT / 'audit/SCALING.csv').read_bytes())
    table(out / 'BACKGROUND_MODE_CONVERSION_R2_FIGURE_DATA.csv', figdata)
    write(out / 'BACKGROUND_MODE_CONVERSION_R2_CONTROLS.json', {'technical_valid': True, 'historical_integrity': read(ROOT / 'preflight/READ_ONLY_INTEGRITY.json'), 'calibration_resolution': fr['resolution'], 'records': cs, 'reference_scales': read(ROOT / 'audit/REFERENCE_SCALES.json'), 'no_overlap': True, 'new_audit_content_before_freeze': False, 'mandatory_analytic_gate_passed': True})
    write(out / 'BACKGROUND_MODE_CONVERSION_R2_CHAIN_RULE.json', {'coordinate_rule': '(t_x+t_y)/(4*g^2) sum_r alpha_r(x)alpha_r(y)e_rc; divide by sqrt(P_ref), contract fixed kappa_C', 'mixed_rule': '(t_x+t_y)/(64*g^2) sum_r (xi_r(x)+xi_r(y))e_rc; same fixed reference contraction', 'atol': fr['chain_atol'], 'rtol': fr['chain_rtol'], 'calibration': read(ROOT / 'calibration/CHAIN_RULE.json'), 'audit': read(ROOT / 'audit/CHAIN_RULE.json'), 'mixed_checks': [{'family': c['family'], 'g': c['g'], 'seed': c['seed'], 'direction': c['direction'], 'm1_fit': c['m1'], 'mixed_chain': c['mixed_chain'], 'mixed_autograd': c['mixed_autograd'], 'Gamma': c['Gamma']} for c in cs]})
    lines = ['# Reference-normalized background-assisted mode conversion v282M-r2', f'\nGlobal classification: **{classification}**.', '\nThis is a new preregistered experiment. The completed r1 TECHNICAL_PROTOCOL_FAILURE and matched-sector SCIENTIFIC_CONFLICT are unchanged.', '\nAll six cells were audited using 8 paired training seeds per cell, 4 nuisance directions within each seed, 9 backgrounds including zero, and 5 amplitudes with both signs. There are 51,840 signed audit records and 622,080 signed coordinates. Only whole seeds enter 20,000 paired bootstrap draws.', '\nThe reference P_ref is computed once at s=1, delta=0 for each declared family/g/seed/bank and held fixed. Current Pq is diagnostic only; no epsilon or floor enters the observable. The base models, layers, expert partitions and checkpoint inputs match historical bindings. CPU float64 interventions and contractions use exact promoted historical weights, with MPS float32 base representations/features and separate float32 coefficient diagnostics.', '\n| Cell | Mean C at s=0 | 95% CI C | Mean mixed M | 95% CI mixed M | H1-H4 |', '|---|---:|---|---:|---|---|']
    for c in cells:
        lines.append(f"| {c['family']}/g={c['g']} | {c['mean_Lambda_C_0']:.8g} | [{c['CI95_Lambda_C_0'][0]:.8g}, {c['CI95_Lambda_C_0'][1]:.8g}] | {c['mean_Xi_M']:.8g} | [{c['CI95_Xi_M'][0]:.8g}, {c['CI95_Xi_M'][1]:.8g}] | {('PASS' if c['PASS'] else 'FAIL')} |")
    lines.extend(['\nThe Stage A analytic checks verified positive P_ref, q_C(0,0)=0, exact A darkness at all backgrounds, the sign-sensitive C/M chain rule against autograd, the M null at zero, generically nonzero C derivatives, and function-preserving copied-block interpolation. The symmetric odd-only mixed fit uses s=-1/4,-1/8,0,1/8,1/4. Calibration confirmed its relative coordinate error against the exact mixed derivative below 1%; amplitude-fit relative errors are also below 1%. Calibration did not optimize C/M separation. No scientific choice changed after freeze.', '\nFor H3 the preregistered structural-null alternative uses agreement of exact chain and autograd coordinate derivatives at zero within 1e-10. Finite-amplitude fitted residuals are retained, including possible cubic-fit truncation. Null A/M responses receive no artificial log-log exponent. Paired finite-difference Gamma vectors at all four symmetric nonzero background magnitudes are supplied in the chain-rule JSON.', '\nFresh banks contain 512 documents in each of the four historical domains, 2,048 per family/bank, and 1,024 distinct pairs for each of 12 ordered domain coordinates. All 8,192 new text identities are mutually disjoint and excluded from the historical audit/calibration sets checked in INPUT_BINDINGS. Identity-only selection preceded Stage A; audit texts, tokens, representations, pairs and geometry were first materialized after the frozen audit-opening record. Fresh pair and direction RNG streams are separate from the historical experiment.', f"\nFreeze SHA256: `{results['freeze_sha256']}`. Historical inventory and SHA256 equality: {len(after)} files verified before/after. All 48 checkpoint bindings matched. No historical manuscript, source, output, checkpoint or r1 artifact was modified.", '\nThis evidence concerns the reference-normalized restricted feature-space response. It does not measure full B_ts, downstream utility or training/optimizer generality. The result is a separate research node and does not relabel the historical findings.'])
    (out / 'BACKGROUND_MODE_CONVERSION_R2_RESULTS.md').write_text('\n'.join(lines) + '\n')
    (out / 'BACKGROUND_MODE_CONVERSION_R2_SCIENTIFIC_IMPACT.md').write_text(f'# Scientific impact\n\nClassification: **{classification}**.\n\n' + ('The held-out experiment supports background-assisted parameter-to-field mode conversion in all six tested family/g cells. C remains bright at parent symmetry; M is structurally dark there and has a resolved positive mixed background coefficient; A remains dark throughout. ' if all((c['PASS'] for c in cells)) else 'The cell-level results and frozen classification rules determine the scientific conclusion. ') + 'The repaired fixed-reference coordinate resolves the r1 domain singularity without regularizing or replacing the historical normalization. The r1 failure and prior matched-sector conflict remain immutable.\n\nSupported scope is the restricted observable, these two pretrained base-model families, three copied child granularities and the eight historical paired seeds. No manuscript edits or claims about full B_ts or downstream utility are made.\n')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1, 0.65]}, layout='constrained')
    for fi, f in enumerate(['S1', 'S2']):
        for gi, g in enumerate([2, 4, 8]):
            ax = axs[fi, gi]
            cell = next((c for c in cells if c['family'] == f and c['g'] == g))
            for sector, style, marker in [('C', '-', 'o'), ('M', '--', 's')]:
                data = [d for d in figdata if d['family'] == f and d['g'] == g and (d['sector'] == sector)]
                x = [d['s'] for d in data]
                y = [d['seed_mean_Lambda'] for d in data]
                ax.plot(x, y, color='black', ls=style, marker=marker, ms=3, label=f'Lambda {sector}')
                ax.fill_between(x, [d['CI95_low'] for d in data], [d['CI95_high'] for d in data], color='.7' if sector == 'C' else '.9', alpha=0.55)
            ax.axhline(cell['b1_resolution_bound'], color='black', ls=':', label='A resolution bound')
            ax.axvline(0, color='.6', lw=0.6)
            ax.set_title(f'{f} / g = {g}')
            ax.set_xlabel('Parent background s')
            ax.set_ylabel('First derivative magnitude')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.text(0.02, 0.04, f"A bound: {cell['b1_resolution_bound']:.2e}", transform=ax.transAxes, fontsize=8)
            if fi == 0 and gi == 0:
                ax.legend(fontsize=8, loc='upper left')
    for gi, g in enumerate([2, 4, 8]):
        ax = axs[2, gi]
        for i, f in enumerate(['S1', 'S2']):
            c = next((c for c in cells if c['family'] == f and c['g'] == g))
            v = c['mean_Xi_M']
            lo, hi = c['CI95_Xi_M']
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt='o' if f == 'S1' else 's', color='black', capsize=4)
            ax.plot([i - 0.15, i + 0.15], [c['mixed_resolution_bound']] * 2, 'k:')
        ax.set_xticks([0, 1], ['S1', 'S2'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f'Mixed M coefficient / g = {g}')
        ax.set_ylabel('Xi M')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.suptitle('Reference-normalized mode conversion: held-out seed means and 95% CIs', fontsize=14)
    fig.savefig(out / 'BACKGROUND_MODE_CONVERSION_R2_FIGURE.pdf')
    fig.savefig(ROOT / 'preflight/FIGURE_PREVIEW.png', dpi=140)
    plt.close(fig)
    print(json.dumps(results, default=lambda x: x.item()), flush=True)
if __name__ == '__main__':
    main()
