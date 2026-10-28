# Generic non-mode-aligned local dynamics — recovered historical model-facing source

Recovery status: **RECOVERED_EXECUTED_SOURCE / PASS**.

Published here:
- `protocol.json`: public-safe protocol/config copy; only the preferred local model path was replaced.
- `scripts/generic_core.py`: exact historical local scientific dependency, byte-identical.
- `scripts/run_generic_local_dynamics.py.part00`
- `scripts/run_generic_local_dynamics.py.part01`
- `scripts/run_generic_local_dynamics.py.part02`

Reconstruct the exact public runner with:

```sh
cat scripts/run_generic_local_dynamics.py.part00 \
    scripts/run_generic_local_dynamics.py.part01 \
    scripts/run_generic_local_dynamics.py.part02 \
  > scripts/run_generic_local_dynamics.py
sha256sum scripts/run_generic_local_dynamics.py
```

Expected runner SHA-256:

`b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744`

Expected Git blob identity of the reconstructed runner:

`ed4a758d4c89c0473cd397e6ceb3bb6fa4a5a0c5`

Chunk Git blob identities:
- part00: `bb3ce3d5a3e34d3e78f6490eec8542a7debe9f65`
- part01: `32a572b95e79b633ff16e79dd69be8da1dd89091`
- part02: `1f14b14832ecec628ec4a06422a7b45b1a3ea7ab`

Historical execution binding:
- protocol ID: `Project1_GenericLocalDynamics_PRX_v1_2026-10-27`
- package manifest SHA-256: `87cd5c6c146ba9fb0d3e4119e68568c3dd2915e55845ea35c54e9374f0113487`
- original delivery ZIP SHA-256: `0416209469638564db16543808390efa0ab63ef47064a33a93a60c5896e20f3d`
- local dependency SHA-256: `91dd65aa0819ef7ccd88bbb561c2990f0a43161e9d212853f836a49a3140d3f0`

The historical recovery audit verified all seven Python files byte-identical to the retained package and bound them through the archived execution log and package manifest. The public repository intentionally does not mirror private F0/operator inputs, binary result arrays, model weights, corpus text, caches, or private path receipts.
