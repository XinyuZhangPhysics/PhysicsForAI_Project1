# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Single audited byte index and payload gateway. No payload diagnostics or reprs."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import re
from common import *

class AccessDenied(RuntimeError):
    pass

def build_index(source, metadata, output):
    byid = {r['document_id']: r for r in metadata}
    index = {}
    with Path(source).open('rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            matches = re.findall(b'"document_id"\\s*:\\s*"([^"\\\\]+)"', line)
            if len(matches) != 1:
                raise RuntimeError('BLIND_INDEX_ID_FAILURE')
            doc = matches[0].decode('ascii')
            if doc not in byid or doc in index:
                raise RuntimeError('BLIND_INDEX_MEMBERSHIP_FAILURE')
            index[doc] = dict(offset=offset, length=len(line), **byid[doc])
    if len(index) != len(byid):
        raise RuntimeError('BLIND_INDEX_COUNT_FAILURE')
    write(output, dict(source_sha256=sha(source), payloads_decoded=0, rows=index))
    return index

class Firewall:

    def __init__(self, source, index, partitions, directory, log, freeze_sha):
        self.source = Path(source)
        self.index = index
        self.directory = Path(directory)
        self.log = Path(log)
        self.freeze_sha = freeze_sha
        self.members = {}
        for bank, rows in partitions.items():
            for r in rows:
                if r['document_id'] in self.members:
                    raise RuntimeError('BANK_OVERLAP')
                self.members[r['document_id']] = (bank, r)
        self.decoded_counts = {'ADAPT': 0, 'HELDOUT': 0, 'RESERVE': 0}

    def _audit(self, doc, arm, access, result, reason, bank):
        append_event(self.log, dict(document_id=doc, bank=bank, arm=arm, access_class=access, result=result, reason=reason))

    def _sealed(self, arm):
        folder = self.directory / 'arms' / arm
        receipt = folder / 'adaptation_receipt.json'
        seal = folder / 'adaptation_receipt.sha256'
        state = folder / 'final_alpha.json'
        try:
            r = read(receipt)
            return sha(receipt) == seal.read_text().split()[0] and r['arm'] == arm and (r['scientific_freeze_sha256'] == self.freeze_sha) and (r['state_sha256'] == sha(state)) and (r['epochs_completed'] == 2) and (r['adapt_documents_processed'] == 448) and (r['optimizer_steps'] == 56) and (r['base_parameter_updates'] == 0) and (r['status'] == 'SEALED_FINAL')
        except (OSError, KeyError, ValueError):
            return False

    def _decode(self, doc, arm, access):
        bank, row = self.members.get(doc, ('UNKNOWN', None))
        reason = None
        if arm not in ARMS:
            reason = 'UNKNOWN_ARM'
        elif bank == 'RESERVE':
            reason = 'RESERVE_ALWAYS_FORBIDDEN'
        elif bank == 'ADAPT' and access != 'adaptation':
            reason = 'ADAPT_WRONG_CODE_PATH'
        elif bank == 'HELDOUT' and access != 'evaluation':
            reason = 'HELDOUT_WRONG_CODE_PATH'
        elif bank == 'HELDOUT' and (not self._sealed(arm)):
            reason = 'ARM_NOT_SEALED'
        elif row is None:
            reason = 'UNKNOWN_DOCUMENT'
        if reason:
            self._audit(doc, arm, access, 'DENIED', reason, bank)
            raise AccessDenied(reason)
        try:
            loc = self.index[doc]
            with self.source.open('rb') as f:
                f.seek(loc['offset'])
                payload = f.read(loc['length'])
            decoded = json.loads(payload)
            ids = decoded['token_ids']
            valid = decoded['document_id'] == doc and decoded['domain'] == row['domain'] and (decoded['token_array_sha256'] == row['token_array_sha256']) and (len(ids) == 256) and all((type(t) is int and 0 <= t < 50304 for t in ids)) and (digest(canonical(ids)) == row['token_array_sha256'])
            if not valid:
                raise RuntimeError('SELECTED_PAYLOAD_INTEGRITY_FAILURE')
        except Exception:
            self._audit(doc, arm, access, 'FAILED', 'PAYLOAD_INTEGRITY_FAILURE', bank)
            raise RuntimeError('SELECTED_PAYLOAD_INTEGRITY_FAILURE') from None
        self.decoded_counts[bank] += 1
        self._audit(doc, arm, access, 'PERMITTED', 'IDENTITY_VALIDATED', bank)
        return ids

    def load_adapt(self, doc, arm):
        return self._decode(doc, arm, 'adaptation')

    def load_heldout(self, doc, arm):
        return self._decode(doc, arm, 'evaluation')

def actual_firewall():
    f = frozen()
    partitions = {}
    for bank, spec in f['manifests'].items():
        p = BUNDLE / spec['file']
        if sha(p) != spec['sha256']:
            raise RuntimeError('MANIFEST_INTEGRITY_FAILURE')
        partitions[bank] = [json.loads(line) for line in p.open()]
    p = BUNDLE / 'blind_byte_index.json'
    if sha(p) != f['blind_index_sha256']:
        raise RuntimeError('INDEX_INTEGRITY_FAILURE')
    index = read(p)
    if sha(FRAME) != index['source_sha256']:
        raise RuntimeError('FRAME_INTEGRITY_FAILURE')
    return (Firewall(FRAME, index['rows'], partitions, BUNDLE, BUNDLE / 'access_events.jsonl', sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json')), partitions)
