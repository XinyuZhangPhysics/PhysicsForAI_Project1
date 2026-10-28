# Expert-Specific Response in Mixture-of-Experts — reference implementation

Clean mathematical implementations for operators retained in the active manuscript.

Run data-independent checks with:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The tests cover expert-index selection, signed mixed pushforward, mixed-Hessian adjointness/exact-discrete growth, and finite-time ordering/cumulant identities.

Removed trained sparse-MoE fixed-routing/tomography/objective/utility programs are absent from this reference layer. The exact executed source for the 2026-10-27 generic-local model-facing run remains pending public recovery and is not replaced by a reconstruction.
