# v318 paper-output mapping

The exact delivered v318 main paper and supplement are the claim reference. See `PAPER_RESULT_MAP.md` for the experiment-by-experiment mapping, `REPRODUCIBILITY_LEVELS.md` for the public/internal boundary, and `MODEL_FACING_RERUNS.md` for the historical execution requirements. Run `python reproduce/reproduce_paper.py` and verify `expected/OUTPUT_SHA256SUMS.txt`. This reconstructs selected frozen numerical outputs; it does not reproduce the full historical model-facing executions or PDF pixels.
