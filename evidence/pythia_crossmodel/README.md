# Pythia-70M cross-model replication evidence

This directory contains the compact frozen outputs used for the paper's Pythia-70M cross-model replication.

- Frozen model-facing source-package SHA-256: `626d5bd2eb37af1f91d94bfec4fcbcb57f57878a977f3b1d1440f15b21aa20d1`.
- Verified results-archive SHA-256: `1ffd29f52fd39fa4a3d5fe21062854325253b2aead964dbd44d552d498b691cf`.

The replication independently remeasures the Pythia layer-3 mixed operator. It tests reproduction of the within-objective local mechanism; it does not transport the numerical SmolLM2 operator into Pythia.

The compact numerical outputs are public here. The frozen model-facing source package identified by the hash above is not redistributed in this repository, so the cross-model result is output-verifiable here but its original model-facing execution is not independently reconstructible from this repository alone.
