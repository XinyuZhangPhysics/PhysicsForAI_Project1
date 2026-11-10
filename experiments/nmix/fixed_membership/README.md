# Fixed-membership routing experiment

This experiment evaluates the same 12-dimensional controls while each token's top-8 expert membership is held at its baseline membership.

Perturbed router logits still pass through the full 64-way softmax. The eight baseline expert indices gather their perturbed full-softmax probabilities without within-support renormalization. Expert radial gains and the original downstream suffix are unchanged.

The helper implementations in `source/` provide the fixed-membership suffix, load-balancing term, and deterministic probability-gather backward. Reference sufficient statistics cover amplitudes 0.01, 0.02 and 0.04.
