# NMIX shared assets

This directory records the fixed identities and coordinate definitions shared by the five mature-OLMoE experiments in the sibling directories.

Set `PROJECT1_OLMOE_MODEL` to a local copy of the exact model revision and `PROJECT1_NMIX_BANK_JSONL` to the exact 448-record bank. The code verifies the model files and bank identity against `config.json` before measurement.

`T0_BANK_SPLIT.tsv`, `T0_CONTRAST_BASIS.tsv`, and `T0_DIRECTIONS.tsv` are compact deterministic experiment inputs. The raw third-party-derived token bank and model weights are not redistributed.
