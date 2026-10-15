# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Synthetic fixtures only; no real Q3 payloads are decoded by these tests."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import contextlib, io, tempfile, unittest
from firewall import *

class Tests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.source = self.root / 'synthetic.jsonl'
        self.meta = []
        self.parts = {}
        for bank in ['ADAPT', 'HELDOUT', 'RESERVE']:
            row = dict(document_id='synthetic_' + bank, domain='dclm', token_array_sha256=digest(canonical([1] * 256)))
            self.meta.append(row)
            self.parts[bank] = [row]
        self.source.write_bytes(b''.join((canonical(dict(**r, token_ids=[1] * 256)) + b'\n' for r in self.meta)))
        self.index = build_index(self.source, self.meta, self.root / 'index.json')
        self.fw = Firewall(self.source, self.index, self.parts, self.root, self.root / 'log.jsonl', 'synthetic_freeze')

    def tearDown(self):
        self.temp.cleanup()

    def seal(self):
        folder = self.root / 'arms/CANDIDATE'
        folder.mkdir(parents=True)
        write(folder / 'final_alpha.json', dict(alpha7=0, alpha15=0))
        write(folder / 'adaptation_receipt.json', dict(arm='CANDIDATE', scientific_freeze_sha256='synthetic_freeze', state_sha256=sha(folder / 'final_alpha.json'), epochs_completed=2, adapt_documents_processed=448, optimizer_steps=56, base_parameter_updates=0, status='SEALED_FINAL'))
        (folder / 'adaptation_receipt.sha256').write_text(sha(folder / 'adaptation_receipt.json') + '\n')

    def test_premature_heldout_denied_without_decode(self):
        with self.assertRaises(AccessDenied):
            self.fw.load_heldout('synthetic_HELDOUT', 'CANDIDATE')
        self.assertEqual(self.fw.decoded_counts['HELDOUT'], 0)

    def test_reserve_denied_even_after_seal_all_access_classes(self):
        self.seal()
        for access in ['adaptation', 'evaluation', 'index', 'reserve']:
            with self.assertRaises(AccessDenied):
                self.fw._decode('synthetic_RESERVE', 'CANDIDATE', access)
        self.assertEqual(self.fw.decoded_counts['RESERVE'], 0)

    def test_adapt_wrong_codepath_and_heldout_optimization_denied(self):
        with self.assertRaises(AccessDenied):
            self.fw.load_heldout('synthetic_ADAPT', 'CANDIDATE')
        self.seal()
        with self.assertRaises(AccessDenied):
            self.fw.load_adapt('synthetic_HELDOUT', 'CANDIDATE')

    def test_correct_adapt_and_post_seal_evaluation(self):
        self.assertEqual(len(self.fw.load_adapt('synthetic_ADAPT', 'CANDIDATE')), 256)
        self.seal()
        self.assertEqual(len(self.fw.load_heldout('synthetic_HELDOUT', 'CANDIDATE')), 256)

    def test_tampered_state_and_receipt_rejected(self):
        self.seal()
        write(self.root / 'arms/CANDIDATE/final_alpha.json', dict(alpha7=1))
        with self.assertRaises(AccessDenied):
            self.fw.load_heldout('synthetic_HELDOUT', 'CANDIDATE')

    def test_domain_and_payload_integrity_failure_redacted(self):
        p = json.loads(self.source.read_bytes().splitlines()[0])
        p['domain'] = 'wiki'
        lines = self.source.read_bytes().splitlines()
        lines[0] = canonical(p)
        self.source.write_bytes(b'\n'.join(lines) + b'\n')
        with self.assertRaisesRegex(RuntimeError, 'SELECTED_PAYLOAD_INTEGRITY_FAILURE'):
            self.fw.load_adapt('synthetic_ADAPT', 'CANDIDATE')

    def test_hash_chain_and_no_stdout_or_payload_in_log(self):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.fw.load_adapt('synthetic_ADAPT', 'CANDIDATE')
            with self.assertRaises(AccessDenied):
                self.fw.load_heldout('synthetic_RESERVE', 'CANDIDATE')
        self.assertEqual(out.getvalue() + err.getvalue(), '')
        prev = '0' * 64
        for i, line in enumerate((self.root / 'log.jsonl').read_text().splitlines()):
            r = json.loads(line)
            h = r.pop('event_sha256')
            self.assertEqual(r['sequence'], i)
            self.assertEqual(r['previous_event_sha256'], prev)
            self.assertEqual(digest(canonical(r)), h)
            self.assertNotIn('token_ids', r)
            prev = h
if __name__ == '__main__':
    unittest.main()
