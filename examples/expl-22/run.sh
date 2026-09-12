#!/bin/bash
# Tensor-train conformer search of caffeine where the candidate evaluation
# is a NATIVE MACE singlepoint via libtorch (in-process C++/TorchScript,
# no Python, no socket).  With [ttconf] sp=true the per-batch candidate
# list goes through the GPU-batched pipeline of crest_sploop.
#
# Requirements:
#   - crest built with libtorch support (see input.toml header)
#   - a TorchScript .pt model; set MODEL below (or edit input.toml)

command -v crest >/dev/null 2>&1 || {
  echo >&2 "Cannot find crest binary."
  exit 1
}

MODEL="${MODEL:-/path/to/mace-off23-lammps.pt}"
if [[ ! -f "$MODEL" ]]; then
  echo >&2 ""
  echo >&2 "ERROR: TorchScript model not found: $MODEL"
  echo >&2 "Export a MACE-LAMMPS-format TorchScript checkpoint, e.g. with the"
  echo >&2 "export script shipped in the crest-mlip distribution (scripts/export_mace.py),"
  echo >&2 "then re-run:  MODEL=/path/to/model.pt $0"
  echo >&2 ""
  exit 1
fi

# point the input file at the model, then run
sed "s|/path/to/mace-off23-lammps.pt|$MODEL|" input.toml > input.resolved.toml
crest input.resolved.toml
rm -f input.resolved.toml
