# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import re
from dataclasses import asdict, dataclass
from . import MODEL_REPO, TARGETS, StopPilot
from .reports import write_json
PATTERN = re.compile('^step([0-9]+)-tokens([0-9]+)B$')

@dataclass(frozen=True)
class Revision:
    revision: str
    step: int
    tokens_B: int
    revision_commit_sha: str | None = None

def parse_revision(name, commit=None):
    match = PATTERN.fullmatch(name)
    return Revision(name, int(match[1]), int(match[2]), commit) if match else None

def select_revisions(revisions, targets=TARGETS):
    result = []
    for target in targets:
        selected = min(revisions, key=lambda r: (abs(r.tokens_B - target), r.tokens_B, r.step, r.revision), default=None)
        mismatch = abs(selected.tokens_B - target) / target if selected else None
        result.append(dict(target_tokens_B=target, **asdict(selected) if selected else dict(revision=None, step=None, tokens_B=None, revision_commit_sha=None), relative_mismatch=mismatch, status='RESOLVED' if mismatch is not None and mismatch <= 0.05 else 'UNRESOLVED'))
    return result

def resolve(out, meta):
    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(MODEL_REPO, repo_type='model')
    groups = {group: [dict(name=r.name, ref=r.ref, target_commit=r.target_commit) for r in getattr(refs, group)] for group in ('branches', 'tags', 'converts')}
    parsed = [parse_revision(r.name, r.target_commit) for r in refs.branches]
    chosen = select_revisions([r for r in parsed if r is not None])
    write_json(out / 'all_revisions.json', dict(refs=groups), meta)
    write_json(out / 'pilot_revision_map.json', dict(selection_rule='nearest_tokens; ties lower_tokens,step,name', selected=chosen, resolver_ran_before_probe=True), meta)
    if any((r['status'] != 'RESOLVED' for r in chosen)):
        raise StopPilot('One or more frozen targets have no branch within 5%.')
    return chosen

def load_map(path):
    import json
    data = json.loads(path.read_text())
    if data.get('model_repo') != MODEL_REPO or not data.get('resolver_ran_before_probe'):
        raise StopPilot('Missing or incompatible frozen revision map.')
    selected = data['selected']
    if [r['target_tokens_B'] for r in selected] != list(TARGETS):
        raise StopPilot('Frozen targets changed.')
    for row in selected:
        parsed = parse_revision(row['revision'], row['revision_commit_sha'])
        if not parsed or row['status'] != 'RESOLVED' or (not row['revision_commit_sha']) or (parsed.tokens_B != row['tokens_B']) or (parsed.step != row['step']) or (not re.fullmatch('[0-9a-f]{40}', row['revision_commit_sha'])) or (abs(parsed.tokens_B - row['target_tokens_B']) / row['target_tokens_B'] > 0.05) or (row['relative_mismatch'] != abs(parsed.tokens_B - row['target_tokens_B']) / row['target_tokens_B']):
            raise StopPilot('Unresolved or corrupted frozen revision map.')
    all_path = path.parent / 'all_revisions.json'
    if not all_path.is_file():
        raise StopPilot('Complete frozen revision list is missing.')
    all_refs = json.loads(all_path.read_text())
    if all_refs.get('model_repo') != MODEL_REPO:
        raise StopPilot('Complete revision list has incompatible model_repo.')
    parsed = [parse_revision(r['name'], r['target_commit']) for r in all_refs['refs']['branches']]
    expected = select_revisions([r for r in parsed if r is not None])
    if selected != expected:
        raise StopPilot('Selected checkpoints no longer match the frozen nearest-branch resolver.')
    return selected
