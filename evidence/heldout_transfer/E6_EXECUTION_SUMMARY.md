# E6 execution summary

E6_PROSPECTIVE_TRANSFER_NOT_ESTABLISHED

| Primary metric | Point | 95% stability interval |
|---|---:|---:|
| E_source | 0.068788841 | [0.0672592424, 0.0697918993] |
| C_source | 0.99763324 | [0.997563825, 0.997738071] |
| R_mix | 1.38203676 | [1.363216, 1.40673316] |

Primary bank-median R_mix (T0–T5): 1.37748313, 1.38659039, 1.42136919, 1.37030353, 1.39209714, 1.35612848.
Blind replication bank-median R_mix (T6–T7): 1.40165704, 1.43055261.

Bank 7, h=8 endpoint medians: E_source=0.027020555; R_mix=1.44216576; C_source=0.999634926.

Post-primary target oracle across primary banks (median of six bank medians): E_target_oracle=7.28459877e-07; fraction of squared source residual removed=0.999999999891048.

Nonlinear bank-gradient evaluations: 184. Scientific elapsed time: 925.170945 seconds.

Verification: package verifier PASS; independent mandatory checklist PASS. Row counts: primary 96, blind replication 32, short-time 72, target oracle 128, short-time oracle 72. Replication model evaluations and target-operator captures were zero at PRIMARY_SEAL.

TECHNICAL_ADAPTATIONS.md: None. Exact environment and assets reused. PREANALYSIS_FREEZE.md and protocol_e6.json are byte-identical to the input package. No manuscript or research record was edited.
