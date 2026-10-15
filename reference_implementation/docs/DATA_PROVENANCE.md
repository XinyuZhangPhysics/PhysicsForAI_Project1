# Data provenance and public boundary

`data/derived/` contains project-owned, compact numerical outputs and anonymized completion receipts. `data/derived/provenance.json` binds selected public files to source hashes. The accompanying private `PAPER_DATA_MANIFEST.tsv` and source archive manifests map raw assets to cold-archive members; private absolute paths are deliberately omitted here. Original scientific data and model weights remain local and private. Public data are CC BY 4.0; code is MIT. Third-party model/corpus licenses must be checked upstream.

The public release reconstructs paper-level tables and figures from frozen outputs. It does not claim to regenerate those inputs from raw text or weights. The 17 recovered historical source hashes missing from earlier archive indexes are preserved in a separate internal delta. Large archive volumes and that new delta have no authenticated remote-content SHA256 yet. Data archive: DOI pending.
