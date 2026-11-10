# Expert-index basis resampling

This experiment repeats the source-to-heldout measurement in four independently generated centered six-dimensional expert-index subspaces at amplitude 0.02.

The basis seeds are 2026110902, 2026110903, 2026110904 and 2026110905. `source/runtime_config.py` regenerates the four bases deterministically; their expected SHA-256 values are listed in `basis_identities.json`.

`reference/basis_*.json` contains compact sufficient statistics `H`, `Z^T Z`, `Y^T Z`, and `||Y||_F^2`. These are sufficient to reconstruct the full and block-diagonal prediction residuals and all spectrum-matched orientation ensembles without redistributing model caches.
