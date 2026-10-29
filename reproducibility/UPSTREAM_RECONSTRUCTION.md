# Reconstructing the project-owned experimental inputs

The repository does not redistribute third-party model weights or corpus text. All project-owned information needed to identify the reported experimental samples is public here: pinned upstream revisions, exact document/row identities, SHA-256 hashes, pair-selection seeds, training seeds, protocol/configuration files, and historical experiment source.

## Pinned upstream assets

- Dataset: `DKYoon/SlimPajama-6B@b5f90f419b7489cdba26fdbc8c022fcb5562f968`
- S1 model/tokenizer: `EleutherAI/pythia-70m@a39f36b100fe8a5377810d56c3f4789b9c53ac42`
- S2 / formation model/tokenizer: `HuggingFaceTB/SmolLM2-135M@28e66ca6931668447a3bac213f23d990ad3b0e2b`

The manuscript layer choices are recorded in `configs/models.json`.

## Exact sample identities

`reproducibility/upstream_selection/S1/` contains the frozen S1 calibration/audit bank manifests and eight training-seed sequence manifests. Historical pair manifests are deterministic functions of those bank identities and the published pair seeds in `DATA_MANIFEST_HASHES.json`; the public S1 bank-building source implements the pairing algorithm.

`reproducibility/upstream_selection/S2/` contains the frozen S2 calibration/audit document identities, eight training-seed document manifests, and `BANKS_INDEX.json` with the dataset/model revisions, selection salt, pair seeds, training seeds, counts, and historical manifest hashes. Historical pair identities are deterministic functions of the bank identities and published pair seeds; the public S2 `build_banks.py` implements the pairing algorithm.

These files contain identities and hashes, not third-party text. After obtaining the pinned SlimPajama revision, run

```bash
python reproducibility/verify_upstream_manifests.py \
  --data /path/to/SlimPajama-6B
```

to verify every referenced upstream row against its published SHA-256 hash.

One way to obtain the third-party assets is:

```bash
hf download --repo-type dataset DKYoon/SlimPajama-6B \
  --revision b5f90f419b7489cdba26fdbc8c022fcb5562f968 \
  --include 'data/train-*.parquet' --local-dir /path/to/SlimPajama-6B

hf download EleutherAI/pythia-70m \
  --revision a39f36b100fe8a5377810d56c3f4789b9c53ac42 \
  --local-dir /path/to/pythia-70m

hf download HuggingFaceTB/SmolLM2-135M \
  --revision 28e66ca6931668447a3bac213f23d990ad3b0e2b \
  --local-dir /path/to/SmolLM2-135M
```

## Rebuilding derived inputs and rerunning experiments

The historical source under `experiments/` performs tokenization, representation capture, mixed-operator construction, training, and trajectory measurement. Configure `configs/paths.template.json` through `PROJECT1_PATHS_FILE` to point at the locally downloaded upstream assets and writable work/output locations.

For S1, the public materialization and training-stream scripts verify source-document hashes and token-sequence hashes while rebuilding the banks/streams. For S2, the public bank/materialization scripts rebuild the prospective banks and streams from the pinned corpus and the frozen selection metadata. Formation, generic-local, and finite-time arrays—including F0 captures, mixed operators/spectra, checkpoints, and trajectory tables—are generated intermediate outputs of the corresponding public preflight/experiment stages, rather than additional data that must be obtained from the author.

The compact evidence under `evidence/`, provenance mappings under `reproducibility/`, and public tests provide checks of the reconstructed pipeline. Third-party assets remain subject to their original licenses and access terms.
