from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parents[1] / "reference_implementation" / "src"
sys.path.insert(0, str(SRC))

from moe_private_response import native_tomography as nt

with np.load(ROOT / "checkpoint_41_tomography.npz") as baseline,      np.load(ROOT / "checkpoint_5033_tomography.npz") as terminal,      np.load(ROOT / "control_basis.npz") as basis:
    expected = json.loads((ROOT / "expected_gates.json").read_text())
    scalars = json.loads((ROOT / "response_scalars.json").read_text())

    checked = 0
    for item, target in zip(scalars, expected):
        layer = int(item["layer"])
        probe = item["probe"]
        controls = basis[f"L{layer}_controls_exact_common_projected"]
        H = baseline[f"L{layer}_H"]
        np.testing.assert_allclose(controls @ controls.T, H, rtol=2e-10, atol=2e-13)

        prefix = f"L{layer}_{probe}_J"
        B0 = baseline[prefix + "_physical"]
        B1 = terminal[prefix + "_physical"]
        scale = float(item["Q5033_mode1"]) / float(item["Q41_mode1"])
        gate = nt.scalar_gate(B0, B1, scale)
        np.testing.assert_allclose(gate["scale_error"], target["epsilon_scale"], rtol=2e-10, atol=2e-13)
        np.testing.assert_allclose(gate["shape_cosine"], target["shape_cosine"], rtol=2e-10, atol=2e-13)

        for archive, label in ((baseline, "baseline"), (terminal, "terminal")):
            for half in (0, 1):
                g = nt.split_gate(
                    archive[prefix + f"_half{half}_physical"],
                    archive[prefix + f"_half{1-half}_physical"],
                )
                assert np.isfinite(g["shape_cosine"])
                assert np.isfinite(g["relative_deff_difference"])
            checked += 1

print(f"public v1 longitudinal compact verification PASS ({checked} operator checks)")
