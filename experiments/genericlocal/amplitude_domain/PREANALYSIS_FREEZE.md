# PREANALYSIS FREEZE — E5 amplitude-domain locality scan

This file is frozen before any new non-unit-amplitude outcome is inspected.

## 1. Scientific question

For the already frozen generic non-mode-aligned directions of Section IV, determine the amplitude domain over which the frozen local update map remains a controlled approximation to the nonlinear transformer update.

The exact duplicated-block identity has the common-amplitude scaling

\[
\Delta f(aX,aU)=O(a^2)+O(a^4),
\]

so the first displayed relative correction to the local update is expected to begin at

\[
e_1(a)=O(a^2)
\]

once numerical error is subdominant. The scan tests this scaling and measures an operational locality radius.

## 2. Frozen parent assets

No new direction, bank, layer, model, restoring control, learning rate, or mixed operator is chosen.

Reuse exactly:

- model revision `28e66ca6931668447a3bac213f23d990ad3b0e2b`;
- layer 20 (zero based);
- F0 bank and P2H spectrum bound by the parent protocol;
- \(\mu=0.017010007119361258\);
- \(\eta_X=\eta_U=11.757784614467004\);
- \(\epsilon=-0.08,+0.08\);
- the eight PCG64 direction seeds `2026102701` through `2026102708`.

The parent generic-local runner SHA-256 must be

`b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744`.

## 3. Amplitude grid

Define the existing central-norm reference amplitude

\[
A_\mathrm{align}=a_{0,\mathrm{aligned}}\sqrt{\mu}
\]

and freeze

\[
q\in\{1,2,4,8,16,32,64,128,256\},\qquad A=qA_\mathrm{align}.
\]

No extra amplitude may be inserted after outcomes.

The historical generic-local `large` amplitude is exactly \(q=1\). The former `small` amplitude \(q=1/2\) is not rerun because it is already dominated by the float32/numerical floor and is not needed to measure the nonlinear scaling window.

## 4. Measurement

For every side, direction, and amplitude, perform exactly one nonlinear F0 gradient update and one frozen-map update from identical initial bytes.

Primary observable:

\[
e_1=
\frac{\|\Delta z_{\rm obs}-\Delta z_{\rm lin}\|_{\rm c}}
{\|\Delta z_{\rm lin}\|_{\rm c}+\delta}.
\]

Also retain the absolute central residual, increment cosine, F0 NLL, router-field magnitude, and exact tanh-remainder diagnostics.

Expected scientific gradient evaluations:

\[
2\times8\times9=144.
\]

## 5. Technical gates

Before scientific cells:

1. exact parent runner hash check;
2. exact model revision / architecture check;
3. F0 and spectrum hashes checked by the parent runner inputs;
4. top singular-pair matrix-free action residual below the existing parent tolerance;
5. all eight frozen directions generated from the same seeds.

A failure here is a technical stop.

A nonfinite result that occurs only at a large scientific amplitude after the technical gates is an observed out-of-domain outcome, not grounds for retuning or adding intermediate amplitudes.

## 6. Primary scaling test

The scaling window is fixed before outcomes:

\[
q\in\{4,8,16,32\}.
\]

For each of the stable and unstable sides separately, compute the median \(e_1(q)\) over the eight directions and fit

\[
\log \operatorname{median} e_1(q)=c+\alpha\log q
\]

by ordinary least squares on these four points.

`LOCALITY_SCALING_RESOLVED` requires:

- median \(e_1(q=1)<5\times10^{-6}\) on both sides;
- complete finite data through \(q=32\) for all eight directions on both sides;
- \(1.5\le\alpha\le2.5\) independently on both sides.

Otherwise the label is `LOCALITY_SCALING_NOT_RESOLVED`, with the failed condition stated explicitly. No alternate fit window is permitted for the primary decision.

Direction-wise slopes over the same fixed window are secondary diagnostics only.

## 7. Operational locality radius

For each side define

\[
q_{1\%}=\min\{q:\operatorname{median}e_1(q)\ge0.01\},
\]

and similarly \(q_{5\%}\) at 0.05.

If a threshold is not crossed by \(q=256\), report only the lower bound \(q_{1\%}>256\) or \(q_{5\%}>256\). No interpolation is used for the primary radius statement; interpolation may be shown only as a descriptive secondary diagnostic.

## 8. Interpretation

A resolved quadratic window supports the analytically implied local effective description and identifies the amplitude range in which higher-order corrections first become measurable above the numerical floor.

A later threshold crossing supplies an operational breakdown scale. Absence of a crossing gives a lower bound on the locality radius over the scanned family.

The experiment does not study long-time saturation, mode selection, top-k support changes, or semantic expert specialization.

## 9. Forbidden after outcomes

No new amplitude, direction, fit window, error threshold, side, horizon, model, seed, bank, precision mode, learning rate, restoring curvature, or normalization may be introduced to rescue the primary label.
