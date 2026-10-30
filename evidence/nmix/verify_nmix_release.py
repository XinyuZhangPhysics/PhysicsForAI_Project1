from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with (ROOT / "NMIX_PROTOCOL.json").open() as f:
    protocol = json.load(f)

with (ROOT / "NMIX_PRIMARY_RESULTS.tsv").open(newline="") as f:
    primary = list(csv.DictReader(f, delimiter="\t"))
with (ROOT / "NMIX_DOMAIN_RESULTS.tsv").open(newline="") as f:
    domains = list(csv.DictReader(f, delimiter="\t"))
with (ROOT / "NMIX_ADJUDICATION.tsv").open(newline="") as f:
    adjudication = list(csv.DictReader(f, delimiter="\t"))

assert protocol["status"] == "GRADE_A_DELETE_NATIVE_QUALIFICATION"
assert protocol["model"]["revision"] == "b99743a400844e532bec7ce4575d97a3a39dcd4d"
assert protocol["bank"]["archived_jsonl_sha256"] == "0dd17c1eaa7f3106d92080d99b0d1386cf3b9cdae0edea75e2012321a8d09d2a"
assert protocol["coordinates"]["primary_amplitude"] == 0.02
assert protocol["bootstrap"]["replicates"] == 10000
assert protocol["bootstrap"]["seed"] == 2026103003

cross = {float(r["amplitude"]): r for r in primary if r["scope"] == "cross_bank"}
assert set(cross) == {0.01, 0.02, 0.04}
for a, row in cross.items():
    ef = float(row["E_full"])
    en = float(row["E_no_mixed"])
    rr = float(row["R"])
    assert math.isclose(ef / en, rr, rel_tol=3e-6, abs_tol=3e-7), (a, ef / en, rr)
    assert int(row["improved_direction_pairs"]) == 12

p = cross[0.02]
assert float(p["E_full"]) <= protocol["grade_A"]["E_full_max"]
assert float(p["R"]) <= protocol["grade_A"]["R_max"]
assert float(p["median_full_cosine"]) >= protocol["grade_A"]["median_full_cosine_min"]
assert float(p["bootstrap_R_upper95"]) < protocol["grade_A"]["bootstrap_R_upper_strict_max"]
assert all(float(cross[a]["R"]) < 1.0 for a in (0.01, 0.04))
assert len(domains) == 7 and all(float(r["R"]) < 1.0 for r in domains)

lookup = {(r["grade"], r["check"]): r for r in adjudication}
assert lookup[("A", "E_full")]["result"] == "PASS"
assert lookup[("A", "R")]["result"] == "PASS"
assert lookup[("A+", "median_support_turnover")]["result"] == "FAIL"
assert lookup[("A+", "p95_support_turnover")]["result"] == "FAIL"

print("NMIX compact release consistency: PASS")
print("Primary R:", p["R"])
print("Primary paired-direction 95% CI:", p["bootstrap_R_lower95"], p["bootstrap_R_upper95"])
print("Grade A: PASS; Grade A+: FAIL (support-turnover criteria only)")
