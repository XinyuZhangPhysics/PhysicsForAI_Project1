# Reproducibility levels

1. **Public code-level reproducibility.** The clean operator implementations and data-independent unit tests are public and can be run from a fresh clone.
2. **Paper-level numerical reconstruction with supplied inputs.** The reconstruction scripts are public, while the project-owned derived numerical inputs are not bundled. They may be obtained from the author on reasonable request and supplied through `PROJECT1_DATA_DIR`.
3. **Full model-facing rerun.** This additionally requires retained raw banks/captures, historically executed sources and environments, and third-party model/corpus assets obtained under their original licenses.
4. **Provenance-only historical records.** Some archived source records establish provenance but do not by themselves establish exact execution identity.

The source-only public release does not contain frozen result arrays or output checksums.
