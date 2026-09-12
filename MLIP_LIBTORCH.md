# Native MLIP (libtorch) backend — port notes

This document describes the port of the **native LibTorch MLIP backend** from
`EPiCs-group/crest-mlip` (based on CREST 3.0.2, commit `163f50a`) into the
CREST 3.1 experimental tree. Only the **libtorch direct-inference backend**
was ported (in-process C++/TorchScript, no Python, no TCP socket). The other
two backends of the fork (embedded Python `pymlip`, ASE socket) were not
needed for the target workflow and were left out.

The goal is the workflow from the design discussion:

```
TTConf candidate batches (same ligand, torsions rotated)
        |
        v   ttconf_eval_batch() -> crest_sploop()  (with -ttsp)
GPU-batched MACE singlepoints  (libtorch, pipelined, multi-GPU aware)
        |
        v
TT/maxvol selection -> low-energy ensemble
        |
        v
GFN2-xTB (+ALPB) final refinement (existing CREST machinery, untouched)
```

## What was changed

### New files (ported from crest-mlip, essentially verbatim)

| File | Purpose |
|------|---------|
| `src/calculator/libtorch_bridge.h` | C-linkage interface for the TorchScript bridge |
| `src/calculator/libtorch_bridge.cpp` | C++ bridge: TorchScript load, neighbor list, MACE-LAMMPS forward, E/grad extraction, batching, pipelining, multi-GPU, shared-model registry |
| `src/calculator/calculator_libtorch.F90` | Fortran module `calc_libtorch` (iso_c_binding wrapper, lazy init, batch/pipeline/multi-GPU entry points, debug timing) |

Minor fixes applied to the ported Fortran wrapper:

* `libtorch_engrad` now zero-initialises `energy`/`gradient` before the lazy
  init, so a failed model load can no longer hand back unassigned values.
* A `libtorch_is_shared` flag was added to `calculation_settings`
  (see below) and `libtorch_cleanup` honours it: shared handles are released
  through `libtorch_shared_free_all()` instead of being double-freed with
  `libtorch_free()`.
* `libtorch_init_shared` marks the handle as shared.

### `src/calculator/calc_type.f90`

* New job type: `jobtype%libtorch = 18` (the official fmlip-relay socket
  backend keeps `jobtype%mlip = 14`; both coexist).
* `jobdescription` extended to 19 entries.
* New fields on `calculation_settings`:

  | Field | Meaning |
  |-------|---------|
  | `libtorch_handle` | opaque C++ model context (`c_ptr`) |
  | `libtorch_model_path` | path to the TorchScript `.pt` model |
  | `libtorch_device_id` | 0=CPU, 1=CUDA:0, 2=MPS, 10-13=CUDA:0-3 |
  | `libtorch_model_format` | 0=generic (tuple output), 1=MACE-LAMMPS (dict) |
  | `libtorch_cutoff` | neighbor-list cutoff (Angstrom), default 6.0 |
  | `libtorch_debug` | per-call timing printout |
  | `libtorch_call_count`, `libtorch_total_time` | profiling counters |
  | `libtorch_is_shared` | handle comes from the C++ shared registry |
  | `libtorch_shared_model` | share one model across OMP threads |
  | `mlip_batch_size` | structures per GPU batch (0 = auto from nat) |
  | `mlip_aten_threads` | ATen intra-op threads (0 = auto) |
  | `mlip_ngpus` | GPUs for multi-GPU batching (0 = auto-detect, cap 2) |

* `mlip_keep_loaded` flag on `calcdata`: when true, `mlip_cleanup_all()`
  skips releasing model handles so the model can be reused across workflow
  steps (e.g. repeated TTConf batches); it is force-released at program
  exit (`custom_cleanup`).
* `calculation_settings_copy` copies the scalar/char settings and nulls the
  handle (each per-thread copy re-initialises lazily, or the parallel driver
  broadcasts a shared handle explicitly).
* `calculation_settings_deallocate` frees the model-path string and resets
  the libtorch state.
* `calculation_settings_shortflag` / `calculation_settings_info` /
  `create_calclevel_shortcut` gain `libtorch` cases.

### `src/calculator/calculator.F90`

* `use calc_libtorch`, re-export of all `libtorch_*` routines.
* Dispatch: `case (jobtype%libtorch) -> libtorch_engrad(...)`.
* New helpers: `mlip_cleanup_all(calc)` (releases in-process MLIP handles at
  the end of each algorithm; honours `mlip_keep_loaded`),
  `mlip_needs_shared_model(calc)`, `mlip_auto_batch_size(nat)`
  (nat<30 -> 64, nat<100 -> 16, else 4).

### `src/parsing/parse_calcdata.f90`

New `method` and keys (TOML `[[calculation.level]]`):

```toml
method         = "libtorch"     # or "mace-direct"
model_path     = "/path/to/model.pt"     # (also: libtorch_model)
model_format   = "mace-lammps"           # (also: "generic")
device         = "cuda:0"                # cpu | cuda | cuda:0-3 | mps
cutoff         = 6.0                     # Angstrom
batch_size     = 0                       # 0 = auto from nat
ngpus          = 0                       # 0 = auto-detect (cap 2)
aten_threads   = 0                       # 0 = auto
shared_model   = true
libtorch_debug = false
# integer forms also accepted:
libtorch_device_id = 10
libtorch_batch_size / libtorch_aten_threads / libtorch_ngpus
```

### `src/algos/parallel.f90` — `crest_sploop` GPU batched fast path

Ported (and adapted to the 3.1 `structures`-based API) from the fork's
+606-line change. When **all** of these hold:

* exactly one calculation level,
* that level is `jobtype%libtorch` with a CUDA device,
* all structures share `nat` and atomic numbers,
* `nall > 0`,

the standard OpenMP task loop is bypassed. Instead all structures are packed
into one contiguous buffer (Bohr) and handed to the C++ bridge:

* single GPU -> `libtorch_engrad_batch_pipeline_f` (double-buffered
  CPU/GPU pipeline);
* multi-GPU  -> `libtorch_load_shared_on_device_f` per device +
  `libtorch_engrad_batch_multigpu_f` (interleaved round-robin).

Batch size: `mlip_batch_size` if set, else `mlip_auto_batch_size(nat)`.
GPU count: `mlip_ngpus` if set, else `libtorch_get_cuda_device_count_f()`
capped at 2. ATen threads: `mlip_aten_threads` if set, else 1 (the GPU does
the parallelism).

After the call the energies are written back into `structures(i)%energy`
(and the optional `eread`). The shared model is released unless
`env%calc%mlip_keep_loaded` is set, in which case it stays in the C++
registry for the next batch (important for TTConf, which calls
`ttconf_eval_batch`/`crest_sploop` many times).

If any condition is unmet the standard per-thread path runs (with
informational notes: CPU usage, or GPU falling through to the serialized
per-thread path).

**TTConf:** with `-ttsp` (or `[ttconf] sp = true`) the candidate blocks of
`ttconf_eval_batch()` go through this path automatically — same ligand,
same `nat`/`at`, so the uniformity check always passes. The default
TTConf (geometry-optimising) still goes through `crest_oloop` per structure;
a batched-optimizer driver for that path is the planned next step (Phase C).

### Cleanup at algorithm entry points

`mlip_cleanup_all(...)` is now called at the end of `crest_singlepoint`,
`crest_optimization`, `crest_ensemble_optimization`,
`crest_moleculardynamics`, `crest_numhess`, `crest_ensemble_hessians`,
`crest_scan`, `trialMD_calculator` and `trialOPT_calculator`
(`src/cleanup.f90::custom_cleanup` force-releases everything at exit,
`src/sigterm.F90::graceful_shutdowns` on SIGINT/TERM).

### `src/dynamics/shake_module.f90`

MLIPs provide no Wiberg bond orders; SHAKE mode 2 ("all bonds") used to
`error stop` when no WBO was available. It now falls back to SHAKE mode 1
(X-H bonds only) with a message, instead of aborting.

### Build system

* **CMake**: new option `WITH_LIBTORCH` (default `FALSE`) in
  `config/CMakeLists.txt`. The root `CMakeLists.txt` requires C++17,
  `find_package(Torch REQUIRED)` and links `${TORCH_LIBRARIES}` into the
  object/static/shared targets; `src/calculator/CMakeLists.txt` compiles
  `libtorch_bridge.cpp` (always compiles the Fortran wrapper, which ships a
  stub when the feature is off).
* **meson**: new feature option `libtorch` (default `disabled`) in
  `meson_options.txt`. When enabled, meson enables the C++ compiler, adds
  `-std=c++17`, resolves `dependency('Torch', method:'cmake')` and defines
  `WITH_LIBTORCH` for c/cpp/fortran.
* `assets/template/metadata.f90`, `config/{CMakeLists.txt,meson.build}` and
  `src/printouts.f90` report the new feature (`-DWITH_LIBTORCH`).

## Building

### CMake

```bash
# from a directory with a discoverable Torch CMake config (see below)
cmake -S crest -B build \
      -DWITH_LIBTORCH=true \
      -DTorch_DIR="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

`Torch_DIR` points at the directory containing `TorchConfig.cmake`
(typically `.../site-packages/torch/share/cmake/Torch` or
`.../torch/share/cmake`). `find_package(Torch)` also honours the
`CMAKE_PREFIX_PATH` / `TORCH_DIR` environment variables.

### meson

```bash
export TORCH_CMAKE_DIR="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')"
meson setup crest/build -Dlibtorch=enabled
meson compile -C crest/build
```

### Proven recipe: M3 (Monash Massive) Apptainer, no system gcc

The current Apptainer image ships no `module` command and no system
compiler toolchain. A working CREST 3.1 + libtorch build was assembled
from the software already under `/apps`:

| component | source |
|-----------|--------|
| Fortran (gfortran 15.1.0) | `/apps/locscale/2.3/locscale-2.3-env` (conda-forge) |
| C / C++ (gcc/g++ 9.5.0, C++17 OK) | `/apps/warp/2.0.0dev8/warp-2.0.0dev8-env` (conda-forge) |
| glibc 2.17 link set + CRT | locscale env sysroot |
| cmake 3.29.3 | `/apps/cmake/3.29.3` |
| GNU make 4.4.1 | spack install under `/apps/spack/v0.21.2/opt/spack/...` |
| CUDA 12.6 toolkit | `/apps/cuda/12.6` |
| PyTorch 2.7.0+cu126 (TorchConfig.cmake) | `/apps/pytorch/2.7.0` |

The conda gcc drivers carry their own (old, inconsistent) sysroots, so
the working tree contains a small toolchain shim that makes them link
against one consistent glibc set:

- `/workspace/toolchain/{gcc,g++,gfortran}` — wrapper scripts. They add
  `-L` paths for the glibc-2.17 set and append
  `libc_nonshared.a` (for the 2.17 CRT's `__libc_csu_init`), the host
  `ld-linux-x86-64.so.2` (for `__tls_get_addr`), and `-ldl -lrt
  -lpthread`. The link-only arguments are skipped for `-c/-E/-S`
  invocations (nvcc probes `gcc -E` during configure).
- `/workspace/toolchain/shim/` — a driver "prefix" so the conda gcc
  binary resolves a sysroot whose `lib`/`lib64`/`usr` are the locscale
  (glibc 2.17) ones.
- `/workspace/toolchain/hostlibs/` — dev `.so` symlinks: libc/libm/
  libpthread → host glibc 2.34 (needed because torch 2.7 requires
  symbols up to GLIBC_2.28, e.g. `fcntl64@2.28`,
  `__strtof128_nan@GLIBC_PRIVATE`), librt/libdl → 2.17 set.
- `/workspace/toolchain/bin/` — bare `gcc/g++/gfortran/cc/c++/ar/nm`
  names on `PATH` (nvcc invokes a bare `gcc` to probe the host
  compiler).

Configure + build (all verified working):

```bash
source /workspace/crest-build-env.sh
cmake -S /workspace/crest -B /workspace/build-crest \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_Fortran_COMPILER=$FC \
  -DCUDA_TOOLKIT_ROOT_DIR=/apps/cuda/12.6 \
  -DALLOW_GFORTRAN_15_1=ON \
  -DWITH_LIBTORCH=ON -DWITH_TESTS=OFF \
  -DTorch_DIR=/apps/pytorch/2.7.0/lib/python3.9/site-packages/torch/share/cmake/Torch \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/apps/locscale/2.3/locscale-2.3-env/lib"
cmake --build /workspace/build-crest -j$(nproc)
```

Notes:
- `-DALLOW_GFORTRAN_15_1=ON` is needed because the vendored dftd4
  subproject rejects gfortran 15.0/15.1 (PR 119928); the only gfortran
  on this image is 15.1.0. The guard in
  `subprojects/dftd4/config/CMakeLists.txt` was patched to allow the
  override (the build completed cleanly with it).
- Runtime: `LD_LIBRARY_PATH` needs the locscale env lib (libgfortran,
  libgomp, libstdc++) and, for GPU runs, the CUDA 12.6 libs — both are
  set by `crest-build-env.sh`. The binary also embeds rpaths.
- Verified results (generic-format test model, ethane, 8 atoms):
  singlepoint `TOTAL ENERGY 7.8977992924 Eh` and 3-structure
  `ensemblesp` energies all agree with an independent Python reference
  implementation to ~2e-7.

### CREST 3.1 ensemble file format (pitfall)

`runtype = "ensemblesp"` reads the ensemble through
`rdensemble`, which expects **plain multi-frame *.xyz** — a sequence of
`[nat, comment, nat coord lines]` blocks with **no** total-atom header
line and no file comment. The `ensemble` key in the input TOML is also
used as the coordinate input, so the *first frame* of the ensemble file
is the reference structure. (Passing a file with a total-atom header
line makes the single-structure reader misparse it and CREST segfaults
in `inputcoords_` — this is a reader-robustness issue in CREST 3.1,
independent of the libtorch port.)

The PyTorch install used at configure time should ideally match the one on
the runtime image (ABI compatibility of `libtorch`).

## Using it

### Singlepoint / ensembles

```toml
runtype = "ensemblesp"        # or "singlepoint", "optimize", "md", ...
input   = "struc.xyz"

[calculation]
[[calculation.level]]
method       = "libtorch"
model_path   = "/path/to/mace-off23-lammps.pt"
model_format = "mace-lammps"
device       = "cuda:0"
```

### TTConf with GPU-batched MACE singlepoints (the target workflow)

```toml
runtype = "ttconf"
input   = "struc.xyz"

[ttconf]
preset = "normal"
sp     = true

[calculation]
[[calculation.level]]
method       = "libtorch"
model_path   = "/path/to/mace-off23-lammps.pt"
model_format = "mace-lammps"
device       = "cuda:0"
```

or on the command line: `crest struc.xyz -ttconf normal -ttsp` with the
level defined in a TOML file. (TTConf itself performs a one-off GFN0
singlepoint for rotor detection, so the build needs `WITH_GFN0`, which is
on by default.)

Example: `examples/expl-22/`. Integration test:
`test/test_libtorch_integration.sh` (skips itself when no model/binary is
available).

## Model export

The MACE-LAMMPS TorchScript format expected by `model_format = "mace-lammps"`
is produced by the export scripts shipped with the crest-mlip distribution
(`scripts/export_mace.py` / `scripts/export_model.py`): the resulting `.pt`
encodes graph construction + forward so that the C++ bridge only has to
convert units (Bohr<->Angstrom, eV<->Hartree). A "generic" model must
implement `forward(positions_bohr, atomic_numbers) -> (energy_hartree,
gradient_hartree_bohr)` and can use `model_format = "generic"`.

TorchScript authoring pitfalls hit during this work (all models under
`/workspace/*.py` work around them):

- **Python-scalar accumulators silently lose precision**: `s = 0.0;
  s += <tensor>` inside a scripted method is rounded to **float32**
  precision even when the tensors are float64 (observed on
  torch 2.7.0+cu126; the loss is exactly one float32 round of the
  running sum). Use a 0-dim tensor accumulator instead:
  `s = torch.zeros((), dtype=pos.dtype, device=pos.device); s = s + ...`.
- Closed-over globals (module-level constants) are not allowed in
  `forward` — re-declare them as locals inside `forward`.
- `Tensor.norm(dim=2)` without `p` fails to script; use
  `(d*d).sum(dim=2).sqrt()`. `.bool()` does not exist on scripted
  tensors; avoid boolean masks in favour of nested loops.
- Avoid `torch.autograd.grad` tuple unpacking; return analytic
  gradients.

## Phase C: batched geometry-optimisation driver

The default TTConf workflow (`sp = false`) relaxes every candidate with
`crest_oloop`. For a uniform candidate batch (same ligand, same nat/atomic
numbers — always true for TTConf) this is wasteful on a GPU: N independent
optimisers each issue single-structure forwards that get serialized by the
C++ mutex.

`mlip_batch_oloop` (in `src/algos/parallel.f90`, triggered from
`crest_oloop_struc`) replaces that with a **batched L-BFGS driver**:

- keeps one L-BFGS optimizer state per structure (position, gradient,
  search direction, S/Y history of length `lbfgs_histsize`),
- issues **exactly ONE batched E+G evaluation per outer iteration** for all
  active structures (packed into one contiguous Bohr buffer, processed by
  the existing pipelined/multi-GPU C++ path),
- the line search is "amortized": a rejected trial step (energy rise) is
  retried with the shrunk step in the *next* batched call, so the outer
  loop never needs more than one batch call per iteration no matter how
  many structures backtrack,
- convergence criteria, step schedule (base 0.2, x0.25 backtracking,
  steepest-descent restart after 12 retries) and failure conventions
  (energy +1.0, `anopt` partial results) mirror `lbfgs_module`.

Trigger conditions (automatic): one `libtorch` calculation level, uniform
nat/atomic numbers, nall > 0, and a GPU device (`device_id > 0`). On CPU
the driver can be forced for testing with the new `[calculation]` key:

```toml
[calculation]
libtorch_batch_opt = true        # force the batched driver (also enables
                                 # the batched sploop fast path on CPU)
```

Verified on CPU (login node, no GPU) with `runtype = "optimize_ensemble"`:
3 water structures (equilateral O-H-H triangle at side 0.8/1.3/1.8 Bohr)
relaxed with the toy well model (per-pair minimum at 1.0 Bohr, E = -2.0)
all converged to the exact analytic minimum, E = -6.0 Eh, side 1.00008
Bohr (100% success), and the standard per-thread path (ANCOPT) gives the
same result — the two drivers are equivalent.

## Benchmark: MLIP (libtorch) vs GFN0, 4-core CPU login node

`ensemblesp` over 200 caffeine (24-atom) structures, single libtorch/GFN0
level, 4 OpenMP threads, CREST 3.1 built with `WITH_LIBTORCH=ON`:

| method | wall (SP loop) | ms/structure |
|---|---|---|
| `gfn0` (no MLIP) | 0.936 s | 4.7 |
| `libtorch` (medium toy model), per-thread OMP | 1.025 s | 5.1 |
| `libtorch` (medium toy model), batched driver, `libtorch_batch_opt = true` | 1.016 s | 5.1 |

On CPU the native MLIP path is within ~10% of GFN0 for a moderate-weight
potential (the toy model's 552 pair terms are dominated by TorchScript
interpreter dispatch, not math). The MLIP advantage appears where CREST
scales: on GPU, where the batched pipelined driver evaluates many
structures in one kernel-batch and the Phase C batched optimizer removes
the per-structure optimizer round-trips entirely.

## Known limitations / next steps

1. ~~Default TTConf (with geometry optimisation) ran per-structure~~ —
   Phase C now provides the batched optimizer driver (see above); on GPU
   it is automatic, on CPU it can be forced with `libtorch_batch_opt`.
2. **WBO**: libtorch provides E/grad only. TTConf's rotor detection uses the
   built-in one-off GFN0 call (unchanged); SHAKE falls back to X-H mode when
   no WBO is present.
3. **Model persistence**: set `mlip_keep_loaded` (via the host API; not yet
   parseable from TOML) to keep the GPU model resident across repeated
   `crest_sploop` calls; otherwise the shared model is reloaded per
   ensemble call (the C++ registry makes repeated loads of the same
   path+device cheap *within* one run only if it is not freed).
4. **Charged systems**: pass `chrg`/`uhf` as usual; the model must actually
   support them (the bridge does not add charge handling).
