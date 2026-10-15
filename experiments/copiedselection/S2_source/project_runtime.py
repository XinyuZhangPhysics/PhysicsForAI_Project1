# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Project the registered 32-run workload from measured preflight components."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np

def main(runtime_path: Path, s1_evaluator: Path) -> None:
    runtime = json.loads(runtime_path.read_text())
    assert runtime['status'] == 'PREFLIGHT_ONLY_COMPLETE' and runtime['benchmark_tokens'] == 100000
    streams = json.loads((runtime_path.parent / 'banks/TRAINING_STREAMS.json').read_text())
    assert streams['status'] == 'FUTURE_SCIENTIFIC_STREAMS_FROZEN_PRE_SCIENCE'
    spec = importlib.util.spec_from_file_location('frozen_evaluate_s1', s1_evaluator)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(2026110296)
    indices = rng.integers(0, 8, size=(20000, 8), dtype=np.int16)
    values = rng.uniform(0.1, 2.0, size=(8, 3))
    t = time.perf_counter()
    for _ in range(3):
        _ = module.bootstrap(values, indices)
    bootstrap_seconds = time.perf_counter() - t
    response = runtime['response_pass']
    training = runtime['training_loop_wall_seconds'] * (32 * 256000 / 100000)
    setup = runtime['setup_seconds'] * 32
    data_preparation = streams['elapsed_seconds'] + runtime['bank_construction_seconds'] * 2
    representation = response['representation_seconds'] * 2
    geometry = response['geometry_seconds'] * 8
    response_passes = 128 + 104
    per_checkpoint_response = response['router_response_seconds'] * response_passes
    data_load = runtime['data_load_seconds'] * 32 + runtime['response_bank_data_load_seconds'] * 2
    measured_component_total = training + setup + data_preparation + representation + geometry + per_checkpoint_response + data_load + bootstrap_seconds
    headroom_factor = 2.0
    projected = measured_component_total * headroom_factor
    runtime['projection'] = {'scientific_runs': 32, 'tokens_per_run': 256000, 'total_training_tokens': 8192000, 'registered_checkpoints_per_run': 4, 'calibration_response_passes': 128, 'audit_response_passes': 104, 'whole_seed_bootstrap_replicates': 20000, 'synthetic_bootstrap_seconds': bootstrap_seconds, 'components_seconds': {'training_including_checkpoints': training, 'model_setup': setup, 'data_preparation_measured_streams_plus_two_bank_proxy': data_preparation, 'representation_extraction_two_banks': representation, 'geometry_eight_arm_bank_combinations': geometry, 'checkpoint_response_232_passes': per_checkpoint_response, 'data_load': data_load, 'bootstrap': bootstrap_seconds}, 'measured_component_total_seconds': measured_component_total, 'conservative_headroom_factor': headroom_factor, 'conservative_end_to_end_seconds': projected, 'conservative_end_to_end_hours': projected / 3600, 'resource_gate_upper_hours': 3.0, 'resource_gate_pass': projected / 3600 <= 3.0, 'historical_21877_tokens_per_second_used': False}
    runtime_path.write_text(json.dumps(runtime, indent=2) + '\n')
    print(json.dumps({'measured_component_total_hours': measured_component_total / 3600, 'conservative_projected_hours': projected / 3600, 'resource_gate_pass': projected / 3600 <= 3.0}))
if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
