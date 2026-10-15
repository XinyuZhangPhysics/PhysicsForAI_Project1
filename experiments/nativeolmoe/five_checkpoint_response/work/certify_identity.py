# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Exactly three fixed documents, 41B only, failing document exactly three repeats."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import csv, gc, time, traceback
CASE_ORDER = ((0, 1), (215, 1), (216, 1), (216, 2), (216, 3))
ATOL = 1e-10
RTOL = 1e-05

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def residual(a, b):
    ab = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
    norm = float(np.linalg.norm(b.astype(np.float64)))
    return (ab, ab / max(norm, 1e-30))

def main():
    e = engine('T1')
    torch = e.torch
    meta = dict(e.META, scope='technical certification only', documents=[0, 215, 216], tokens_B=41, checkpoint=e.CP[41], layers=[7, 15], modes=[1, 2, 3, 4, 5, 6], failing_document_repeat_count=3, existing_accuracy={'absolute': ATOL, 'relative': RTOL}, analytic_reference='FP64 evaluation from cached FP32 routing probabilities/weights, BF16 expert outputs cast to FP32, and the same controls represented in FP32; production kernels unchanged', composition_response_dtype='FP32 component sum', objective_reduction_dtype='FP64 CPU', implementation_sources=e.META['implementation_sources'] + [ident(Path(__file__))])
    diagnostic = RUN / 'T1_NUMERICAL_IDENTITY_DIAGNOSTICS.npz'
    add_npz(diagnostic, {'metadata_json': np.array(json.dumps(meta))}, True)
    rows = []
    case_audits = []
    repeat_signatures = {}
    vectors_info = {}
    model = e.load_checkpoint(e.CP[41], e.CACHE)
    try:
        for index, rep in CASE_ORDER:
            ids = torch.tensor(e.ROWS[index]['token_ids'], device='mps').unsqueeze(0)
            caps, kw, last = e.capture(model, ids)
            with torch.no_grad():
                base = e.logits(model, last)
            bh = array_sha(base.detach().cpu().double().numpy())
            ell0 = float(e.loss_from_logits(base, ids))
            for l in (7, 15):
                local = e.make_local(model, caps, l)
                a, ell = e.adjoint(model, caps[l]['layer_output'], l, kw, ids, bh)
                v = e.BASIS[l]
                vr = v.copy()
                vr[:, :64] = 0
                vg = v.copy()
                vg[:, 64:] = 0
                ur = np.stack([e.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vr])
                ug = np.stack([e.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vg])
                uj = np.stack([e.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in v])
                composed = ur + ug
                from project1_r2.hard_routing_diagnostic import fixed_support_reweighted_routes
                routes = fixed_support_reweighted_routes(local, torch.zeros(64, device='mps', dtype=torch.float32))
                ix = routes.indices.cpu().numpy()
                w = routes.weights.cpu().numpy()
                p = routes.probabilities.cpu().numpy()
                f = np.zeros((256, 8, 2048), dtype=np.float32)
                for item in local.contributions:
                    f[item.tokens.cpu().numpy(), item.slots.cpu().numpy()] = item.outputs.float().cpu().numpy()
                wd = w.astype(np.float64)
                fd = f.astype(np.float64)
                vf = v.astype(np.float32).astype(np.float64)
                y = np.einsum('tk,tkh->th', wd, fd, optimize=False)
                analytic_r = []
                analytic_g = []
                for x in vf:
                    gains, bias = (x[:64], x[64:])
                    gain = np.einsum('tk,tkh->th', wd * gains[ix], fd, optimize=False)
                    mean = np.sum(wd * bias[ix], axis=1) if local.normalized else p.astype(np.float64) @ bias
                    router = np.einsum('tk,tkh->th', wd * bias[ix], fd, optimize=False) - mean[:, None] * y
                    analytic_r.append(router)
                    analytic_g.append(gain)
                ar = np.stack(analytic_r)
                ag = np.stack(analytic_g)
                gr = np.einsum('ith,th->i', ur, a, dtype=np.float64, optimize=False)
                gg = np.einsum('ith,th->i', ug, a, dtype=np.float64, optimize=False)
                gjc = gr + gg
                gjd = np.einsum('ith,th->i', uj, a, dtype=np.float64, optimize=False)
                guard_limit = ATOL + RTOL * np.maximum(np.abs(gjd), np.abs(gr) + np.abs(gg))
                gres = np.abs(gjd - gjc)
                key = f'D{index}_REPEAT{rep}_L{l}'
                vals = {'u_R_engine': ur, 'u_G_engine': ug, 'u_J_direct': uj, 'u_R_analytic': ar, 'u_G_analytic': ag, 'u_J_composed': composed, 'shared_adjoint': a, 'shared_support_indices': ix, 'shared_routing_weights': w, 'shared_routing_probabilities': p, 'shared_expert_outputs': f, 'g_R': gr, 'g_G': gg, 'g_J_composed': gjc, 'g_J_direct': gjd, 'direct_objective_absolute_residual': gres, 'old_guard_limit': guard_limit, 'controls_joint': v, 'controls_router': vr, 'controls_gain': vg, 'control_identity_residual': v - vr - vg, 'metadata_json': np.array(json.dumps(dict(meta, document_index=index, repeat=rep, layer=l, source_document=e.ROWS[index], baseline_logits_sha256=bh, baseline_NLL=ell0, normalized_topk=bool(local.normalized), shared_base_state=True, shared_support=True, shared_adjoint=True, token_mask='all 256 response tokens; objective NLL uses positions 0..254 for targets 1..255')))}
                add_npz(diagnostic, {key + '_' + k: a0 for k, a0 in vals.items()})
                signatures = {k: array_sha(vals[k]) for k in ['u_R_engine', 'u_G_engine', 'g_R', 'g_G', 'shared_adjoint', 'shared_support_indices', 'shared_routing_weights', 'shared_expert_outputs']}
                if index == 216:
                    repeat_signatures.setdefault(str(l), []).append(signatures)
                vectors_info[key] = {'prefix': key, 'diagnostic_npz_path': str(diagnostic), 'document_index': index, 'repeat': rep, 'layer': l, 'baseline_NLL': ell0, 'signatures': signatures}
                layer_rows = []
                for j in range(6):
                    dr, rr = residual(ur[j], ar[j])
                    dg, rg = residual(ug[j], ag[j])
                    dj, rj = residual(uj[j], composed[j])
                    control_abs = float(np.max(np.abs(v[j] - vr[j] - vg[j])))
                    row = {'document_index': index, 'repeat': rep, 'layer': l, 'mode': j + 1, 'control_identity_infinity_norm': control_abs, 'gain_block_norm': float(np.linalg.norm(vg[j])), 'router_block_norm': float(np.linalg.norm(vr[j])), 'gain_common_mode_sum': float(vg[j, :64].sum()), 'router_common_mode_sum': float(vr[j, 64:].sum()), 'delta_R_absolute': dr, 'delta_R_relative': rr, 'delta_G_absolute': dg, 'delta_G_relative': rg, 'delta_J_direct_absolute': dj, 'delta_J_direct_relative': rj, 'g_R': float(gr[j]), 'g_G': float(gg[j]), 'g_J_composed': float(gjc[j]), 'g_J_direct': float(gjd[j]), 'direct_objective_absolute_residual': float(gres[j]), 'direct_objective_scale_normalized_residual': float(gres[j] / max(abs(gjd[j]), abs(gr[j]) + abs(gg[j]), 1e-30)), 'old_guard_limit': float(guard_limit[j]), 'old_direct_joint_guard_pass': bool(gres[j] <= guard_limit[j]), 'analytic_router_component_pass': bool(dr <= ATOL + RTOL * np.linalg.norm(ar[j])), 'analytic_gain_component_pass': bool(dg <= ATOL + RTOL * np.linalg.norm(ag[j])), 'shared_base_support_adjoint': True, 'baseline_NLL_exact': ell == ell0, 'protocol_sha256': meta['protocol']['sha256'], 'basis_sha256': meta['basis']['sha256'], 'Bank_T_seal_sha256': meta['Bank_T_seal_sha256'], 'checkpoint_revision': e.CP[41]['revision_commit_sha'], 'provenance_prefix': key}
                    rows.append(row)
                    layer_rows.append(row)
                case_audits.append({'document_index': index, 'repeat': rep, 'layer': l, 'all_values_written_before_gate': True, 'rows': layer_rows})
                print('CERTIFICATION CASE PERSISTED', index, rep, l, 'analytic_R_max', max((r['delta_R_relative'] for r in layer_rows)), 'analytic_G_max', max((r['delta_G_relative'] for r in layer_rows)), 'direct_guard_failures', sum((not r['old_direct_joint_guard_pass'] for r in layer_rows)), flush=True)
                del local, a, ur, ug, uj, composed, ar, ag, vals, fd, f, wd, y
                gc.collect()
                torch.mps.empty_cache()
            del caps, kw, last, base, ids
            gc.collect()
            torch.mps.empty_cache()
    finally:
        del model
        gc.collect()
        torch.mps.empty_cache()
    with (RUN / 'T1_NUMERICAL_IDENTITY_DIAGNOSTICS.csv').open('x', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)
    structural = all((r['control_identity_infinity_norm'] == 0 and r['shared_base_support_adjoint'] and r['baseline_NLL_exact'] for r in rows))
    components = all((r['analytic_router_component_pass'] and r['analytic_gain_component_pass'] for r in rows))
    stable = all((len(s) == 3 and s[0] == s[1] == s[2] for s in repeat_signatures.values()))
    reproduced = any((not r['old_direct_joint_guard_pass'] for r in rows if r['document_index'] == 216 and r['layer'] == 15))
    if not structural or (stable and (not components)):
        label = 'IMPLEMENTATION_BUG_CONFIRMED'
        cert = 'CERT-A'
    elif structural and components and stable and reproduced:
        label = 'DIRECT_JOINT_NUMERICAL_PATH_ONLY'
        cert = 'CERT-B'
    else:
        label = 'COMPONENT_NUMERICS_UNRESOLVED'
        cert = 'CERT-C'
    result = dict(meta, status='COMPLETE', certification_branch=cert, T1_TECHNICAL_CERTIFICATION=label, controls_and_shared_state_pass=structural, analytic_components_pass=components, repeated_components_bitwise_stable=stable, original_direct_guard_failure_reproduced=reproduced, repetition_signatures=repeat_signatures, case_audits=case_audits, diagnostic_case_index=vectors_info, artifacts={n: ident(RUN / n) for n in ['T1_NUMERICAL_IDENTITY_DIAGNOSTICS.npz', 'T1_NUMERICAL_IDENTITY_DIAGNOSTICS.csv']}, guard_relaxed=False, canonical_composition_authorized=cert == 'CERT-B', no_5033_or_later_quantities=True)
    write(RUN / 'T1_NUMERICAL_IDENTITY_CERTIFICATION.json', result)
    write(RUN / 'T1_NUMERICAL_IDENTITY_SEAL.json', dict(meta, status='SEALED', T1_TECHNICAL_CERTIFICATION=label, certification_branch=cert, artifacts={n: ident(RUN / n) for n in ['T1_NUMERICAL_IDENTITY_DIAGNOSTICS.npz', 'T1_NUMERICAL_IDENTITY_DIAGNOSTICS.csv', 'T1_NUMERICAL_IDENTITY_CERTIFICATION.json']}))
    with (RUN / 'T1_NUMERICAL_IDENTITY_SEAL.sha256').open('x') as f:
        f.write(sha(RUN / 'T1_NUMERICAL_IDENTITY_SEAL.json') + '  T1_NUMERICAL_IDENTITY_SEAL.json\n')
    report = ['T1_TECHNICAL_CERTIFICATION = ' + label, 'CERTIFICATION_BRANCH = ' + cert, 'GUARD_TOLERANCE_CHANGED = NO', 'SCIENTIFIC_PROTOCOL_CHANGE = NONE', '5033B_QUANTITIES_COMPUTED = NO', '', 'The fixed certification set was docs 0/215/216, 41B, L7/L15, all six frozen modes. Doc 216 was replayed exactly three times. One shared forward/support and one shared downstream adjoint per layer served all component and direct-joint paths. All numerical vectors and scalars were persisted before certification gates.', '', f'Controls/shared state: {structural}. Analytic router/gain components: {components}. Three-repeat bitwise component stability: {stable}. Original direct-joint scalar guard reproduced at doc216/L15: {reproduced}.', f"Max analytic router relative response error: {max((r['delta_R_relative'] for r in rows)):.9g}; gain: {max((r['delta_G_relative'] for r in rows)):.9g}.", 'Analytic references use the exact cached FP32 softmax state and FP32-represented controls with FP64 evaluation. The unnormalized Top-k case includes all 64 softmax probabilities in the router bias mean, treating nonselected output vectors as zero. No new control, mask, normalization or support was introduced.', '', 'The old 1e-10 + 1e-5 guard remains intact in the audit. A CERT-B authorizes only the composed joint realization and makes the independent direct-joint path non-blocking. Existing caches still require field-by-field auditing and any missing required component fields must be recovered before reuse.', '', 'Full source paths/hashes, protocol/basis/Bank-T hashes, exact checkpoint and environment are recorded in the NPZ metadata and certification JSON. The old blocked run is unchanged.', '']
    with (RUN / 'T1_NUMERICAL_IDENTITY_REPORT.md').open('x') as f:
        f.write('\n'.join(report))
    print('T1_TECHNICAL_CERTIFICATION', label, 'BRANCH', cert, flush=True)
if __name__ == '__main__':
    main()
