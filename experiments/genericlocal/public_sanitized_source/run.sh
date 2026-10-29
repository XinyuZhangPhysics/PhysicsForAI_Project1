#!/bin/bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/results/generic_local_dynamics_2026-10-27"
ARCHIVED_PY="/PATH/TO/compatible-python312/bin/python"
mkdir -p "$OUT"
STAGE="BOOTSTRAP"
PY=""

write_shell_stop_manifest() {
  local code="$1"
  local manifest="$OUT/generic_local_dynamics_manifest.json"
  if [[ -f "$manifest" && -n "${PY:-}" && -x "${PY:-/nonexistent}" ]]; then
    "$PY" - "$manifest" "$STAGE" "$code" <<'PY' || true
import json,sys,datetime
p,stage,code=sys.argv[1],sys.argv[2],int(sys.argv[3])
try: m=json.load(open(p))
except Exception: m={}
if m.get('status')!='STOP': m['previous_status']=m.get('status')
m['status']='STOP'; m['execution_stage']=stage; m['error_type']='SHELL_OR_POSTRUN_FAILURE'; m['error']=f'run.sh exited with status {code} during {stage}'; m['stopped_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); m['scientific_protocol_unchanged']=True
open(p,'w').write(json.dumps(m,indent=2,sort_keys=True,allow_nan=False)+'\n')
PY
  else
    cat > "$manifest" <<EOF
{
  "status": "STOP",
  "scientific_protocol_unchanged": true,
  "execution_stage": "${STAGE}",
  "error_type": "SHELL_OR_PREEXECUTION_FAILURE",
  "error": "run.sh exited with status ${code} during ${STAGE}",
  "gradient_evaluations": 0,
  "expected_gradient_evaluations": 896
}
EOF
  fi
}

on_error() {
  local code="$?"
  trap - ERR
  write_shell_stop_manifest "$code"
  if command -v zip >/dev/null 2>&1; then
    (cd "$ROOT/results" && rm -f generic_local_dynamics_2026-10-27_STOP_BUNDLE.zip && zip -qr generic_local_dynamics_2026-10-27_STOP_BUNDLE.zip generic_local_dynamics_2026-10-27) || true
  fi
  echo "STOP: stage=$STAGE exit=$code" >&2
  exit "$code"
}
trap on_error ERR

if [[ -f "$OUT/generic_local_dynamics_long.csv" || -f "$OUT/generic_local_dynamics_summary.json" || -f "$OUT/generic_local_dynamics_operator.npz" ]]; then
  STAGE="REFUSE_OVERWRITE"
  echo "Refusing to overwrite existing scientific results at $OUT" >&2
  false
fi

exec > >(tee -a "$OUT/RUN_LOG.txt") 2>&1

STAGE="PACKAGE_MANIFEST_SHELL_SEAL"
if command -v shasum >/dev/null 2>&1; then
  ACTUAL_MANIFEST_SHA="$(shasum -a 256 "$ROOT/PACKAGE_MANIFEST.json" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_MANIFEST_SHA="$(sha256sum "$ROOT/PACKAGE_MANIFEST.json" | awk '{print $1}')"
else
  echo "No SHA-256 command available" >&2
  false
fi
EXPECTED_MANIFEST_SHA="$(awk '{print $1}' "$ROOT/PACKAGE_MANIFEST.sha256")"
[[ "$ACTUAL_MANIFEST_SHA" == "$EXPECTED_MANIFEST_SHA" ]] || { echo "PACKAGE_MANIFEST shell seal mismatch" >&2; false; }

STAGE="PYTHON_SELECTION"
if [[ -n "${PFAI_PYTHON:-}" ]]; then
  [[ -x "$PFAI_PYTHON" ]] || { echo "PFAI_PYTHON is not executable: $PFAI_PYTHON" >&2; false; }
  PY="$PFAI_PYTHON"
elif [[ -x "$ARCHIVED_PY" ]]; then
  PY="$ARCHIVED_PY"
else
  echo "Frozen historical Python environment not found. Set PFAI_PYTHON explicitly to a compatible Python 3.12 environment. No system-python fallback is permitted." >&2
  false
fi
printf 'PYTHON=%s\n' "$PY"

STAGE="ENVIRONMENT_VERIFICATION"
"$PY" "$ROOT/scripts/verify_environment.py" --output "$OUT/ENVIRONMENT_VERIFICATION.json"

STAGE="PACKAGE_AND_SCIENCE_SEAL_VERIFICATION"
PYTHONPATH="$ROOT/scripts" "$PY" "$ROOT/scripts/verify_package.py"

STAGE="UNIT_TESTS"
PYTHONPATH="$ROOT/scripts" "$PY" -m unittest discover -s "$ROOT/tests" -v

STAGE="FROZEN_EXPERIMENT"
echo "--- full frozen experiment: operator capture + fail-closed smoke tests + 896 generic full-F0 gradients ---"
PYTHONPATH="$ROOT/scripts" "$PY" "$ROOT/scripts/run_generic_local_dynamics.py" --output "$OUT"

STAGE="INDEPENDENT_RESULT_RECURRENCE_VERIFICATION"
PYTHONPATH="$ROOT/scripts" "$PY" "$ROOT/scripts/verify_results.py" --result-dir "$OUT" --output "$OUT/RESULT_VERIFICATION.json"

STAGE="FINALIZE_RESULT_MANIFEST"
PYTHONPATH="$ROOT/scripts" "$PY" "$ROOT/scripts/finalize_manifest.py" --result-dir "$OUT"

STAGE="FINAL_COMPLETENESS_CHECK"
"$PY" - "$OUT/generic_local_dynamics_manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['status']=='COMPLETE',m
assert m['gradient_evaluations']==896,m['gradient_evaluations']
assert m['expected_gradient_evaluations']==896
assert m['postrun_recurrence_verification']=='PASS'
assert not m['failures'],m['failures']
print('FINAL_COMPLETENESS_CHECK PASS')
PY

STAGE="RESULT_BUNDLE"
(
  cd "$ROOT/results"
  rm -f generic_local_dynamics_2026-10-27_RESULT_BUNDLE.zip
  zip -qr generic_local_dynamics_2026-10-27_RESULT_BUNDLE.zip generic_local_dynamics_2026-10-27
)
trap - ERR
STAGE="COMPLETE"
echo "RESULT_BUNDLE=$ROOT/results/generic_local_dynamics_2026-10-27_RESULT_BUNDLE.zip"
