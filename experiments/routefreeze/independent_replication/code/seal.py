# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Sealing utility: source tree, artefacts, report.  Usage:

    python code/seal.py source
    python code/seal.py artefacts
    python code/seal.py report reports/PRIMARY_REPORT.md
    python code/seal.py verify
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from u4_common import PATHS, TREE, json_dump, now_utc, sha256_file, source_tree_digest

def _manifest(paths) -> dict:
    ent = []
    for pattern in paths:
        for p in sorted(TREE.glob(pattern)):
            if p.is_file():
                ent.append((str(p.relative_to(TREE)), sha256_file(p), p.stat().st_size))
    return ent

def cmd_source() -> None:
    dig = source_tree_digest(TREE, 'code')
    tests = source_tree_digest(TREE, 'tests')
    lines = [f'# SOURCE_HASHES — written {now_utc()}', '']
    for rel, d, sz in dig['files']:
        lines.append(f'{d}  {sz:>9d}  {rel}')
    for rel, d, sz in tests['files']:
        lines.append(f'{d}  {sz:>9d}  {rel}')
    import hashlib
    hb = hashlib.sha256()
    for rel, d, sz in dig['files'] + tests['files']:
        hb.update(f'{rel}\x00{d}\x00{sz}\n'.encode())
    lines += ['', f"CODE_FILES={dig['n_files']}", f"TEST_FILES={tests['n_files']}", f"TREE_DIGEST_SHA256={dig['TREE_DIGEST_SHA256']}", f'CODE_PLUS_TESTS_DIGEST_SHA256={hb.hexdigest()}', '# scope: code/ and tests/ only (the .venv and raw artefacts are excluded)']
    (PATHS['provenance'] / 'SOURCE_HASHES.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_dump({'written_utc': now_utc(), **{k: v for k, v in dig.items()}}, PATHS['provenance'] / 'source_tree_digest.json')
    print(f"sealed {dig['n_files']} code files, TREE_DIGEST_SHA256={dig['TREE_DIGEST_SHA256']}")

def cmd_artefacts() -> None:
    ent = _manifest(['raw/**/*', 'figures/**/*', 'logs/**/*'])
    lines = [f'# ARTEFACT_HASHES — written {now_utc()}', '']
    for rel, d, sz in ent:
        lines.append(f'{d}  {sz:>12d}  {rel}')
    import hashlib
    h = hashlib.sha256()
    for rel, d, sz in ent:
        h.update(f'{rel}\x00{d}\x00{sz}\n'.encode())
    lines += ['', f'ARTEFACT_FILES={len(ent)}', f'ARTEFACT_DIGEST_SHA256={h.hexdigest()}']
    (PATHS['provenance'] / 'ARTEFACT_HASHES.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'sealed {len(ent)} artefacts, ARTEFACT_DIGEST_SHA256={h.hexdigest()}')

def cmd_report(rel: str) -> None:
    p = TREE / rel
    dig = source_tree_digest(TREE, 'code')
    tests = source_tree_digest(TREE, 'tests')
    art = PATHS['provenance'] / 'ARTEFACT_HASHES.txt'
    art_line = ''
    if art.exists():
        for ln in art.read_text(encoding='utf-8').splitlines():
            if ln.startswith('ARTEFACT_DIGEST_SHA256='):
                art_line = ln.split('=', 1)[1]
    lines = [f'# REPORT_SEAL — written {now_utc()}', f'report_path={rel}', f'report_sha256={sha256_file(p)}', f'report_bytes={p.stat().st_size}', f"source_tree_digest={dig['TREE_DIGEST_SHA256']}", f"source_files={dig['n_files']}", f"tests_tree_digest={tests['TREE_DIGEST_SHA256']}", f"tests_files={tests['n_files']}", f'artefact_digest={art_line}', 'sealed_before_unblinding=TRUE']
    (PATHS['provenance'] / 'REPORT_SEAL.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

def cmd_verify() -> None:
    dig = source_tree_digest(TREE, 'code')
    old = PATHS['provenance'] / 'SOURCE_HASHES.txt'
    drift = 'NONE'
    if old.exists():
        txt = old.read_text(encoding='utf-8')
        prev = [ln for ln in txt.splitlines() if ln.startswith('TREE_DIGEST_SHA256=')]
        if prev and prev[0].split('=', 1)[1] != dig['TREE_DIGEST_SHA256']:
            drift = prev[0].split('=', 1)[1]
    print(f"current_source_tree_digest={dig['TREE_DIGEST_SHA256']}")
    print(f'drift={drift}')
    ent = _manifest(['raw/**/*', 'figures/**/*', 'logs/**/*'])
    missing, mismatch, checked = ([], [], 0)
    am = PATHS['provenance'] / 'ARTEFACT_HASHES.txt'
    if am.exists():
        recorded = {}
        for ln in am.read_text(encoding='utf-8').splitlines():
            parts = ln.split('  ', 2)
            if len(parts) == 3 and len(parts[0]) == 64:
                recorded[parts[2]] = parts[0]
        on_disk = {rel: d for rel, d, _ in ent}
        for rel, d in recorded.items():
            checked += 1
            if rel not in on_disk:
                missing.append(rel)
            elif on_disk[rel] != d:
                mismatch.append(rel)
    print(f'artefacts_checked={checked} missing={missing} mismatches={mismatch}')
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    if cmd == 'source':
        cmd_source()
    elif cmd == 'artefacts':
        cmd_artefacts()
    elif cmd == 'report':
        cmd_report(sys.argv[2])
    else:
        cmd_verify()
