# Reproducibility and paper-output mapping — 2026-10-23

The manuscript is the public claim reference. The repository is source-only.

1. Run the data-independent operator/unit tests with
   `python -m unittest discover -s tests -v`.
2. For paper-level numerical reconstruction, obtain the project-owned derived-data bundle from the author and set `PROJECT1_DATA_DIR` before running `python reproduce/reproduce_paper.py`.
3. Full historical model-facing reruns additionally require pinned third-party models/corpora, retained raw inputs, and the historical execution environment.

See `PAPER_RESULT_MAP.md` for the claim-to-code mapping, `REPRODUCIBILITY_LEVELS.md` for the public/private boundary, and `MODEL_FACING_RERUNS.md` for model-facing requirements. No experimental outputs are bundled in the current public tree.
