"""Explicit path configuration for sanitized historical experiment scripts."""
from pathlib import Path
import json, os

def resolve_path(key):
    filename=os.environ.get('PROJECT1_PATHS_FILE')
    if not filename:
        raise RuntimeError('Set PROJECT1_PATHS_FILE to a configured copy of configs/paths.template.json before running historical experiments.')
    mapping=json.loads(Path(filename).read_text())
    value=mapping.get(key)
    if isinstance(value,dict):value=value.get('value')
    if not isinstance(value,str) or not value or value=='REQUIRED':
        raise RuntimeError('Configure the input/model/writable-output path '+key)
    return os.path.expandvars(os.path.expanduser(value))
