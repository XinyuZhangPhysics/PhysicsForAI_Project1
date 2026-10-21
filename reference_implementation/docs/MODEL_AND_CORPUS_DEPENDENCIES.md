# External model and corpus dependencies

Read `configs/paper_models.json` for exact recovered revisions: SmolLM2-135M formation revision `28e66ca6931668447a3bac213f23d990ad3b0e2b`; OLMoE 41B/146B/503B/1740B/5033B checkpoint revisions and layers L7/L15. Router selection uses Pythia-70m S1 layer 3 and SmolLM2-135M S2 layer 19; their exact revisions must be read from retained historical seals and are not guessed here.

The technical companion dated 2026-10-21 pins native route-freeze OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5` and SlimPajama-6B bank revision `00683529288a1d2fe7ec3cf852b5e2449d27e472`. Copied training, formation F0 and finite-time pools use SlimPajama-6B revision `b5f90f419b7489cdba26fdbc8c022fcb5562f968`. The native six-control bank is 448 documents from seven domains of OLMoE-mix-0924 revision `efe872cfd4bcd7b9233fdb1f7289b3295f120f3a`.

Model weights, third-party code and corpus text are excluded from both public code and data packages. Obtain them from their upstream repositories under their own license/terms; verify pinned revisions before a model-facing rerun.
