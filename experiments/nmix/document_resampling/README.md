# Document-assignment resampling

This experiment repeats the source-to-heldout measurement for four alternative within-domain 32/32 source/heldout assignments of the same 448 documents.

Seeds are 2026110911–2026110914. `source/generate_splits.py` reproduces the assignments from the shared bank manifest. The measurement uses amplitude 0.02 and otherwise the same model, objective, coordinates and routing semantics as the source-to-heldout experiment.

`reference/split_*.json` contains compact sufficient statistics for independent residual/null reconstruction.
