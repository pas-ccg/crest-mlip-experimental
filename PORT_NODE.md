# Native MLIP (LibTorch) Backend — Port Notes

## 1. Overview

This document covers the native LibTorch MLIP backend, [EPiCs-group/crest-mlip](https://github.com/EPiCs-group/crest-mlip), added to the experimental CREST 3.1 branch. The backend runs TorchScript models directly through C++.

The port additionally includes batched geometry optimisation. It uses the existing CREST calculator interface and works alongside the other calculation methods.

The main use case is TTConf. It generates conformers of the same molecule, allowing their energies and gradients to be evaluated in batches.

```text
TTConf candidate generation
           |
           v
Batched MLIP singlepoints
           |
           v
TT/maxvol selection
           |
           v
Batched MLIP optimisation (optional)
           |
           v
Final refinement with GFN2-xTB
```

The port covers the native LibTorch backend. The embedded Python and socket-based MLIP backends were left out because the target workflow uses direct inference.

## 2. Files and Calculator Integration

### 2.1 Native bridge

Three files provide the LibTorch interface:

| File | Purpose |
|---|---|
| `src/calculator/libtorch_bridge.h` | C interface declarations |
| `src/calculator/libtorch_bridge.cpp` | Model loading, neighbour lists, inference, batching and GPU handling |
| `src/calculator/calculator_libtorch.F90` | Fortran bindings and model management |

The C++ bridge loads TorchScript models, prepares their inputs, runs inference, and returns energies and gradients. It also manages shared model instances and distributes batches across GPUs.

The Fortran wrapper uses `iso_c_binding` to call the bridge. Model instances are stored as opaque C pointers.

Two changes were made to model handling.

First, `libtorch_engrad` now initialises the energy and gradient outputs before attempting to load a model. If initialisation fails, the routine returns initialised values.

Second, shared model handles are tracked separately. `libtorch_cleanup` releases shared handles through the shared-model registry, which prevents them from being freed twice.

### 2.2 Calculation settings

The native backend has its own calculation type, separate from the existing socket-based MLIP backend.

The following settings were added to `calculation_settings`:

| Field | Purpose |
|---|---|
| `libtorch_handle` | Pointer to the C++ model context |
| `libtorch_model_path` | TorchScript model path |
| `libtorch_device_id` | CPU, CUDA or MPS device selection |
| `libtorch_model_format` | Generic or MACE-LAMMPS model interface |
| `libtorch_cutoff` | Neighbour-list cutoff, default 6.0 Å |
| `libtorch_debug` | Per-call timing output |
| `libtorch_call_count` | Number of inference calls |
| `libtorch_total_time` | Accumulated inference time |
| `libtorch_is_shared` | Whether the handle belongs to the shared registry |
| `libtorch_shared_model` | Whether model instances are shared between threads |
| `mlip_batch_size` | Structures per batch |
| `mlip_aten_threads` | ATen intra-op thread count |
| `mlip_ngpus` | Number of GPUs used |

The calculation dispatcher calls `libtorch_engrad` when this calculation type is selected.

When calculation settings are copied for parallel execution, scalar and string settings are copied, while the native handle is reset. The receiving thread either initialises its own model or uses a shared handle assigned by the parallel driver.

Model paths and native handles are released during cleanup.

### 2.3 Model lifetime

Model loading is lazy. A model is loaded when the first calculation requires it.

Normally, models are released when an algorithm finishes. The `mlip_keep_loaded` flag allows the calling code to retain a model across several calculation steps.

This matters for TTConf, which may call the ensemble evaluation routine repeatedly. Keeping the model resident avoids repeated initialisation and weight loading between batches.

Shared model instances are managed through the C++ registry. Final cleanup releases any remaining instances when CREST exits.

## 3. Configuration

The backend is selected using `method = "libtorch"`. The alias `mace-direct` is also accepted.

```toml
[calculation]

[[calculation.level]]
method = "libtorch"
model_path = "/path/to/model.pt"
model_format = "mace-lammps"
device = "cuda:0"

cutoff = 6.0
batch_size = 0
ngpus = 0
aten_threads = 0

shared_model = true
libtorch_debug = false
```

`model_format` accepts `mace-lammps` or `generic`.

The `device` setting accepts `cpu`, `cuda`, `cuda:0` through `cuda:3`, and `mps`, subject to runtime support. Integer device identifiers are also supported.

A value of `0` for `batch_size`, `ngpus`, or `aten_threads` selects the corresponding automatic setting.

The current automatic batch sizes are:

| Atom count | Batch size |
|---|---|
| `nat < 30` | 64 |
| `30 <= nat < 100` | 16 |
| `nat >= 100` | 4 |

These defaults use atom count as a simple estimate of batch memory requirements. Larger models may need smaller batches, and the values can be overridden in the configuration.

Automatic GPU selection detects available CUDA devices and currently limits the count to two. The GPU batch path uses one ATen thread by default.

## 4. Batched Singlepoint Evaluation

### 4.1 Batch selection

The batch path is implemented in `crest_sploop`.

It is used when:

- There is exactly one calculation level.
- The level uses the native LibTorch backend.
- The selected device is CUDA.
- The ensemble contains at least one structure.
- All structures have the same atom count and atomic-number sequence.

When these conditions hold, the driver packs the coordinates into a contiguous buffer in Bohr and sends the batch to the C++ bridge.

If any condition is unmet, CREST uses the existing per-structure loop.

The CPU batch path can also be enabled explicitly for testing.

### 4.2 Single-GPU execution

On one GPU, the driver calls `libtorch_engrad_batch_pipeline_f`.

The implementation uses a double-buffered pipeline to overlap CPU preparation with GPU work where possible.

The batch size comes from `mlip_batch_size` or the atom-count heuristic described above.

Batching reduces the number of separate inference submissions. The resulting throughput depends on model complexity, batch size and device utilisation.

### 4.3 Multi-GPU execution

For multiple GPUs, the driver loads a shared model instance on each selected device and calls `libtorch_engrad_batch_multigpu_f`.

Batches are assigned to devices in round-robin order.

After inference, energies and gradients are returned to the corresponding structures. Energy results are written into `structures(i)%energy` and the optional result array.

The model remains loaded between batches when model retention is enabled.

### 4.4 TTConf

TTConf is a suitable workload for this path because its candidate conformers have the same atoms in the same order.

With `-ttsp`, candidate evaluation goes through `ttconf_eval_batch` and `crest_sploop`. The batch path is therefore used automatically when the calculation is configured for LibTorch on CUDA.

Without `-ttsp`, TTConf performs geometry optimisation. That path uses the batched optimiser described in the next section.

TTConf still uses its existing GFN0 singlepoint calculation for rotor detection. The build therefore needs GFN0 support for this workflow.

## 5. Batched Geometry Optimisation

### 5.1 Motivation

The ordinary ensemble optimisation path runs an independent optimiser for each structure. Each optimiser requests energy and gradient evaluations as it proceeds.

On a GPU, this produces many small inference calls even when the structures could be evaluated together. The model spends part of its time handling repeated calls and synchronisation.

The batched driver, `mlip_batch_oloop`, collects the active structures and evaluates them together once per outer iteration.

It is implemented in `src/algos/parallel.f90` and called through `crest_oloop_struc`.

### 5.2 Optimiser state

Each structure keeps its own L-BFGS state:

- Coordinates, energy and gradient.
- Search direction.
- Step size.
- L-BFGS S/Y history.
- Convergence and failure status.

At each iteration, the driver collects the active structures, evaluates their energies and gradients in one batch, and updates each optimiser independently.

Structures that have converged are excluded from later batches.

The optimiser histories remain independent, while model evaluations are grouped into batches.

### 5.3 Line search

The driver uses deferred backtracking.

If a structure's trial step increases its energy, the step is rejected and reduced. The new trial is evaluated in the next outer iteration, together with the other active structures.

Each outer iteration makes one batched energy-and-gradient call, including iterations where some structures are backtracking.

The current step schedule uses:

- Initial step size: 0.2.
- Backtracking factor: 0.25.
- Steepest-descent restart after 12 rejected steps.

The driver follows the existing L-BFGS conventions for history length, convergence and failure reporting.

Deferred backtracking changes the sequence of trial evaluations relative to the ordinary per-structure optimiser. The two implementations may therefore follow different trajectories, particularly on non-convex potential-energy surfaces.

### 5.4 Activation

The driver is selected automatically for compatible LibTorch GPU calculations.

It can be forced on CPU with:

```toml
[calculation]
libtorch_batch_opt = true
```

This also enables the batched singlepoint path on CPU.

The CPU option allows the batch driver to be tested without GPU resources.

## 6. Changes to Existing CREST Behaviour

### 6.1 Calculator cleanup

Cleanup calls were added to the main calculation entry points, including singlepoint evaluation, optimisation, ensemble optimisation, molecular dynamics, Hessian calculations and scans.

These calls release native model resources unless the host calculation has requested model retention.

The shutdown path also releases remaining handles.

### 6.2 SHAKE and bond orders

The LibTorch backend provides energies and gradients but does not supply Wiberg bond orders.

Previously, SHAKE mode 2 could terminate when bond-order information was unavailable.

The modified code falls back to SHAKE mode 1, which constrains X–H bonds only.

The fallback allows the calculation to continue with a smaller set of constraints. Users should account for this difference when interpreting molecular dynamics results.

### 6.3 Charge and spin

The existing `chrg` and `uhf` parameters remain available.

The bridge does not automatically pass these values into the model or modify its predictions to account for charge and spin.

Charged and open-shell systems therefore require a model that supports them through its own interface.

## 7. Building

### 7.1 CMake

The new CMake option is `WITH_LIBTORCH`, disabled by default.

When enabled, the build requires C++17, locates LibTorch using `find_package(Torch REQUIRED)`, compiles the native bridge and links the Torch libraries.

A typical configuration is:

```bash
cmake -S ./crest -B ./build \
  -DWITH_LIBTORCH=ON \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build ./build --parallel
```

If LibTorch is provided by a Python PyTorch installation, its CMake prefix can be obtained using:

```bash
python3 -c \
  'import torch; print(torch.utils.cmake_prefix_path)'
```

`Torch_DIR` may also be set directly to the directory containing `TorchConfig.cmake`.

The compiler, C++ runtime and LibTorch installation must be ABI-compatible. GPU builds also need a compatible CUDA runtime.

The PyTorch installation used to build CREST should match the LibTorch runtime used during execution.

### 7.2 Meson

The Meson feature option is `libtorch`, also disabled by default.

```bash
meson setup ./build ./crest \
  -Dlibtorch=enabled

meson compile -C ./build
```

When enabled, Meson adds C++17 support, resolves the Torch dependency through CMake, and defines `WITH_LIBTORCH`.

The Fortran wrapper is compiled even when LibTorch is disabled. In that case, it provides stub implementations and does not require the C++ backend to be linked.

## 8. Running Calculations

### 8.1 Ensemble singlepoints

```toml
runtype = "ensemblesp"
input = "ensemble.xyz"

[calculation]

[[calculation.level]]
method = "libtorch"
model_path = "/path/to/model.pt"
model_format = "mace-lammps"
device = "cuda:0"
```

For a compatible ensemble, this uses batched GPU inference.

### 8.2 TTConf singlepoints

```toml
runtype = "ttconf"
input = "structure.xyz"

[ttconf]
preset = "normal"
sp = true

[calculation]

[[calculation.level]]
method = "libtorch"
model_path = "/path/to/model.pt"
model_format = "mace-lammps"
device = "cuda:0"
```

The same mode can be requested with `-ttsp` from the command line.

For geometry optimisation, disable `sp` and allow the batched optimiser to run.

### 8.3 Ensemble XYZ format

The `ensemblesp` reader expects a plain multi-frame XYZ file.

Each frame contains an atom count, a comment line and the corresponding atom coordinates:

```text
3
frame 1
O  0.000  0.000  0.000
H  0.758  0.000  0.504
H -0.758  0.000  0.504
3
frame 2
O  0.000  0.000  0.000
H  0.760  0.000  0.510
H -0.760  0.000  0.510
```

There must be no additional file-level header before the first frame.

The ensemble input is also used to obtain the initial reference structure, so the first frame must be valid on its own.

Malformed input may cause errors in the existing CREST reader before the LibTorch backend is called.

## 9. Model Export and Interface

### 9.1 MACE-LAMMPS

The `mace-lammps` format expects a compatible exported TorchScript model.

The bridge prepares the required graph inputs, runs the model and extracts the returned energies and gradients.

Unit conversion is handled at the bridge boundary:

- Bohr to Ångström for coordinates.
- eV to Hartree for energies.
- Corresponding conversion for gradients.

The exported model must use the expected input and output schema. A standard PyTorch checkpoint must be converted to the required TorchScript format.

### 9.2 Generic models

The generic format expects a forward interface equivalent to:

```python
forward(
    positions_bohr,
    atomic_numbers
) -> (
    energy_hartree,
    gradient_hartree_bohr
)
```

The model must return energies and gradients in the required units.

It must also use the gradient sign convention expected by CREST. Returning forces instead of gradients without changing the sign will give incorrect optimisation results.

### 9.3 TorchScript issues

Several TorchScript constraints are worth checking when exporting a model.

**Scalar accumulators**

Python scalar accumulators can cause unintended precision loss when combined with tensors in scripted code.

Use a tensor accumulator when the accumulation must preserve the tensor dtype:

```python
s = torch.zeros(
    (),
    dtype=pos.dtype,
    device=pos.device
)
s = s + value
```

**Module-level constants**

Closed-over Python globals may cause scripting failures. Constants can be declared locally or stored in a TorchScript-compatible form.

**Tensor operations**

Some ordinary PyTorch expressions require changes for TorchScript compatibility.

For example, an explicit norm calculation may be more reliable across supported TorchScript operations:

```python
distance = (d * d).sum(dim=2).sqrt()
```

**Gradients**

The exported model should return gradients using a supported TorchScript interface.

Analytic gradients can avoid restrictions associated with scripting Python-side autograd code. They should be checked against the model's energy output.

## 10. Testing

### 10.1 Functional tests

The implementation was tested with small molecular systems and simple analytic potentials.

The tests covered model loading, batched inference, result ordering, optimisation updates and cleanup.

CPU testing was used to check the batch driver independently of GPU execution.

These tests cover the main calculation paths under controlled conditions. Additional tests are needed for production MLIP workloads.

### 10.2 Batched optimisation

A small ensemble was optimised using an analytic pair potential with a known minimum.

The batched driver converged to the expected geometry and energy for the tested structures. The conventional per-structure optimiser gave consistent results.

The results support the correctness of the batch-state updates and gradient handling for these cases.

Further tests are needed to compare convergence on realistic molecular systems. The line-search schedules differ, and non-convex potential-energy surfaces may produce different optimisation trajectories.

### 10.3 Performance

Preliminary CPU tests showed little difference between per-structure and batched execution for a small toy potential.

For inexpensive models, TorchScript dispatch and batch preparation can account for much of the execution time.

GPU batching reduces separate inference submissions and allows several structures to be evaluated together. The batched optimiser also reduces the number of independent inference sequences.

The performance benefit depends on model size, batch size, molecular size, GPU memory and optimisation behaviour.

Representative GPU benchmarks are still needed to measure throughput and end-to-end optimisation time.

## 11. Limitations

**Model format.** The backend requires a compatible TorchScript model. Other PyTorch checkpoints need conversion.

**Bond orders.** The bridge returns energies and gradients only. Features that require Wiberg bond orders need another calculation method or a fallback.

**Charge and spin.** Support depends on the model interface and training domain.

**Model persistence.** `mlip_keep_loaded` is available through the host API but is not currently exposed in TOML.

**Batch requirements.** Structures must have compatible atom counts and atomic-number ordering. Other ensembles fall back to the standard path.

**Optimisation.** Deferred backtracking changes the order of trial evaluations. Convergence and failure behaviour need further testing on realistic molecular systems.

**GPU performance.** Batch-size defaults and device scheduling use simple heuristics. Larger models and different hardware may require manual tuning.

## 12. Next Steps

The batched optimiser needs broader convergence tests with realistic molecules and MLIP models. The current analytic tests cover the basic implementation, including batch-state updates and gradient handling.

GPU performance should be measured against the ordinary per-structure path using the same model and structures. Useful measurements include singlepoint throughput, optimisation time, peak memory use and scaling across GPUs.

Batch-size selection could account for available GPU memory and model cost. The current heuristic uses atom count alone.

Model persistence and error handling could also be made configurable. Exposing model retention through TOML would simplify repeated TTConf calculations.

Exported model outputs should be compared against their original implementations to check energies, gradients and unit conversion.

## 13. Summary

The port adds native LibTorch inference to CREST 3.1, including batched singlepoint calculations and an L-BFGS-based ensemble optimiser.

The singlepoint path evaluates compatible conformers together. The optimisation path keeps a separate L-BFGS state for each structure and evaluates active structures in batches.

Basic functional tests have passed. Further work will focus on convergence testing with realistic molecular systems and GPU performance measurements.
