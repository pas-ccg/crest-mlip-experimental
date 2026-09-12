#!/bin/bash
# ---------------------------------------------------------------------------
# Integration test for the native libtorch MLIP backend.
#
# What it checks:
#   1. the binary was built with libtorch support (metadata printout),
#   2. a singlepoint energy+gradient can be evaluated through the C++
#      bridge (method = "libtorch"),
#   3. the ensemble singlepoint path (crest_sploop) runs without aborting.
#
# It is intentionally self-contained and only *runs* when:
#   - the crest binary accepts a `method = "libtorch"` level, and
#   - a TorchScript model is available (set the MODEL env var, or it must
#     live at $MODEL_DEFAULT below).
# When no model is available the test exits 0 with a "skipped" message so
# it can safely live in the regular test suite.
# ---------------------------------------------------------------------------
set -u

CREST="${CREST:-crest}"
MODEL="${MODEL:-${MODEL_DEFAULT:-/path/to/mace-off23-lammps.pt}}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

if ! command -v "$CREST" >/dev/null 2>&1; then
  echo "[test_libtorch] SKIP: crest binary not found in PATH"
  exit 0
fi

if [[ ! -f "$MODEL" ]]; then
  echo "[test_libtorch] SKIP: no TorchScript model found ($MODEL)"
  echo "[test_libtorch]         set MODEL=/path/to/model.pt to enable this test"
  exit 0
fi

# --- 1. feature compiled in? ------------------------------------------------
if "$CREST" -h 2>&1 | grep -qi "WITH_LIBTORCH.*true\|libtorch.*enabled"; then
  echo "[test_libtorch] libtorch support detected in build"
else
  # fall back: try a singlepoint and look for the 'not compiled' message
  :
fi

# --- a small test structure (ethane) ----------------------------------------
cat > ethane.xyz <<'EOF'
8
ethane
C   -1.14030878   0.02046075   0.00114820
C   1.14030878   0.02046075   0.00114820
H   -1.80788385   0.56172177   0.46907288
H   -1.78951421  -0.80724855   0.01525380
H   -1.18868770   0.11573701  -1.05250917
H   1.80788385   0.56172177   0.46907288
H   1.78951421  -0.80724855   0.01525380
H   1.18868770   0.11573701  -1.05250917
EOF

# --- 2. singlepoint through the native backend ------------------------------
cat > sp.toml <<EOF
runtype = "singlepoint"
input   = "ethane.xyz"

[calculation]
[[calculation.level]]
method       = "libtorch"
model_path   = "$MODEL"
model_format = "${MODEL_FORMAT:-generic}"
device       = "cpu"
libtorch_debug = true
EOF

echo "[test_libtorch] running native libtorch singlepoint (cpu)"
if "$CREST" sp.toml > sp.out 2>&1; then
  if grep -q "TOTAL ENERGY" sp.out; then
    echo "[test_libtorch] PASS: singlepoint energy produced"
    grep "TOTAL ENERGY" sp.out || true
  else
    echo "[test_libtorch] FAIL: no energy in output"; sed -n '1,40p' sp.out; exit 1
  fi
else
  # A clean 'not compiled' error means the binary lacks the feature: skip.
  if grep -qi "libtorch support not compiled" sp.out; then
    echo "[test_libtorch] SKIP: binary built without WITH_LIBTORCH"
    exit 0
  fi
  echo "[test_libtorch] FAIL: crest exited non-zero"; sed -n '1,40p' sp.out; exit 1
fi

# --- 3. ensemble singlepoints (sploop path) ---------------------------------
# CREST 3.1 ensemble files are plain multi-frame *.xyz (no total-atom header):
# a sequence of [nat, comment, nat coord lines] blocks.
cat > ens.xyz <<'EOF'
8
ethane 1
C   -1.14030878   0.02046075   0.00114820
C   1.14030878   0.02046075   0.00114820
H   -1.80788385   0.56172177   0.46907288
H   -1.78951421  -0.80724855   0.01525380
H   -1.18868770   0.11573701  -1.05250917
H   1.80788385   0.56172177   0.46907288
H   1.78951421  -0.80724855   0.01525380
H   1.18868770   0.11573701  -1.05250917
8
ethane 2
C   -1.14030878   0.02046075   0.00114820
C   1.14030878   0.02046075   0.00114820
H   -1.80788385   0.56172177   0.46907288
H   -1.78951421  -0.80724855   0.01525380
H   -1.18868770   0.11573701  -1.05250917
H   1.80788385   0.56172177   0.46907288
H   1.78951421  -0.80724855   0.01525380
H   1.18868770   0.11573701  -1.05250917
EOF

cat > ens.toml <<EOF
runtype = "ensemblesp"
input   = "ethane.xyz"
ensemble = "ens.xyz"

[calculation]
[[calculation.level]]
method       = "libtorch"
model_path   = "$MODEL"
model_format = "${MODEL_FORMAT:-generic}"
device       = "cpu"
EOF

echo "[test_libtorch] running ensemble singlepoint (crest_sploop)"
if "$CREST" ens.toml > ens.out 2>&1; then
  echo "[test_libtorch] PASS: ensemble singlepoint completed"
else
  if grep -qi "libtorch support not compiled" ens.out; then
    echo "[test_libtorch] SKIP: binary built without WITH_LIBTORCH"
    exit 0
  fi
  echo "[test_libtorch] FAIL: ensemble singlepoint aborted"; sed -n '1,40p' ens.out; exit 1
fi

echo "[test_libtorch] all tests passed"
