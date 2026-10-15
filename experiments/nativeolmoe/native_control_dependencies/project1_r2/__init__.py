# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen R2 engineering pilot; no confirmatory trajectory or utility claims."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
MODEL_REPO = 'allenai/OLMoE-1B-7B-0924'
TARGETS = (41, 1250, 2500, 5033)
LAYERS = (7, 15)
SEED = 20261004
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')

class StopPilot(RuntimeError):
    """A failed or missing prerequisite; never substitute a different experiment."""
