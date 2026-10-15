# Reproducibility levels

1. **Publicly reproducible frozen outputs:** run `python reproduce/reproduce_paper.py` then verify generated output hashes/tolerances. This checks selected numerical claims from compact derived arrays and tables.
2. **Internally reproducible full chain:** requires retained raw maps, gradients, banks and historically executed sources, plus upstream models/corpus. The archive manifest records exact member hashes; remote content verification remains pending for large volumes.
3. **Provenance only / not fully reproducible:** used where execution identity or inputs are unresolved. A matching source SHA alone does not establish which exact script was executed.

No PDF pixel-identity requirement applies to scientific numerical outputs. Full model-facing reruns are not turnkey in this repository.
