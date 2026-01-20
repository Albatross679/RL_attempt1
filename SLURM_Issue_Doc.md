# SLURM Troubleshooting Log

Quick reference for known issues encountered on OSC (Ohio Supercomputer Center) cluster.

## Quick Reference: Common Errors

| Error Pattern | Issue # | Quick Fix |
|---------------|---------|-----------|
| `RuntimeError: Ninja is required` | [#001](#issue-001-ninja-path-not-found) | Export venv/bin to PATH before activation |
| `icc: command line warning` | [#002](#issue-002-icc-deprecation-warning) | Load gcc module, set CC/CXX |
| `Hparam value cannot be changed` | [#003](#issue-003-config-immutability-warnings) | Informational only, not an error |
| `CUDA out of memory` | [#004](#issue-004-cuda-out-of-memory) | Reduce NUM_PARALLEL_ENVS or batch size |
| `ModuleNotFoundError` | [#005](#issue-005-module-import-errors) | Check venv activation, run setup.sh |
| `libimf.so: cannot open` | [#006](#issue-006-intel-runtime-library-missing) | Add Intel runtime lib path to LD_LIBRARY_PATH |
| `GLIBCXX_3.4.32 not found` | [#007](#issue-007-glibcxx-version-mismatch) | Use LD_PRELOAD with GCC 13.2.0 libstdc++ |
| `Deterministic behavior...CUBLAS` | [#008](#issue-008-cublas-deterministic-mode) | Export CUBLAS_WORKSPACE_CONFIG=:4096:8 |

---

## Issue #001: Ninja PATH Not Found

**Job ID:** 43912099
**Status:** RESOLVED
**Date:** 2025-01-19

### Error Message
```
RuntimeError: Ninja is required to load C++ extensions
```

### Root Cause
Ninja IS installed at `.venv/bin/ninja`, but PyTorch's subprocess call couldn't find it because the virtual environment's `bin` directory wasn't in PATH when subprocess spawned.

The issue: `source .venv/bin/activate` only modifies PATH for the current shell. When PyTorch's JIT compilation spawns a subprocess, it inherits PATH but the subprocess doesn't see the venv's bin directory.

### Solution
Export venv bin to PATH **BEFORE** activation:

```bash
# In SLURM script, add BEFORE source activate:
export PATH="${VENV_PATH}/bin:${PATH}"
source "${VENV_PATH}/bin/activate"
```

### Verification
```bash
# Test that ninja is found by subprocess (PyTorch's check)
python3 -c "import subprocess; subprocess.check_output('ninja --version'.split()); print('PASSED')"
```

### Files Modified
- `submit_snake_ppo.slurm` - Added PATH export before venv activation

---

## Issue #002: ICC Deprecation Warning

**Status:** RESOLVED
**Date:** 2025-01-19

### Error Message
```
icc: command line warning #10412: option '-qopenmp' is deprecated
```

### Root Cause
OSC loads Intel compiler (icc) by default. Intel Classic Compiler (icc) is deprecated in favor of Intel oneAPI compilers.

### Solution
Load GCC module and set compiler environment variables:

```bash
module load gcc/13.2.0
export CC=gcc
export CXX=g++
```

### Files Modified
- `submit_snake_ppo.slurm` - Added GCC module load and CC/CXX exports

---

## Issue #003: Config Immutability Warnings

**Status:** NOT A BUG
**Severity:** Informational

### Warning Message
```
WARNING: Hparam value cannot be changed after being used. IGNORED
```

### Explanation
This is expected ALF framework behavior. When gin configuration binds a parameter that has already been accessed, it logs a warning. The training continues normally.

This occurs because:
1. Some default values are read during import
2. Gin config then tries to override them
3. ALF logs a warning but uses the gin-configured value

### Action Required
None. These warnings can be ignored.

---

## Issue #004: CUDA Out of Memory

**Status:** Template for future issues

### Error Message
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Diagnostic Steps
1. Check current GPU memory usage: `nvidia-smi`
2. Check NUM_PARALLEL_ENVS setting
3. Check batch size configuration

### Solutions

**Option 1: Reduce parallel environments**
```bash
export NUM_PARALLEL_ENVS=128  # Default is 256
```

**Option 2: Reduce batch size**
```bash
export MINI_BATCH_SIZE=512  # Default is 1024
```

**Option 3: Enable gradient checkpointing** (if supported by model)

### GPU Memory Guidelines (A100 40GB)
| NUM_PARALLEL_ENVS | Approximate Memory |
|-------------------|-------------------|
| 256 | ~30-35 GB |
| 128 | ~18-22 GB |
| 64 | ~10-14 GB |
| 32 | ~6-8 GB |

---

## Issue #005: Module Import Errors

**Status:** Template for future issues

### Error Message
```
ModuleNotFoundError: No module named 'alf'
ModuleNotFoundError: No module named 'dismech'
```

### Diagnostic Steps
1. Check virtual environment is activated: `which python3`
2. Verify package installation: `pip list | grep alf`
3. Check editable installs: `pip list --editable`

### Solutions

**Reinstall missing package:**
```bash
source .venv/bin/activate
pip install -e ./alf
pip install -e ./dismech-python
```

**Full environment rebuild:**
```bash
rm -rf .venv
./setup.sh
```

---

## Issue #006: Intel Runtime Library Missing

**Job ID:** 43914194
**Status:** RESOLVED
**Date:** 2025-01-19

### Error Message
```
ImportError: libimf.so: cannot open shared object file: No such file or directory
```

### Root Cause
The `cnest` package (used by ALF) was compiled against Intel's libraries. When using GCC instead of ICC (to avoid deprecation warnings), the Intel runtime libraries are no longer loaded by default.

`libimf.so` is the Intel Math Function library, required by `cnest`.

### Solution
Add Intel runtime library path to LD_LIBRARY_PATH (without loading the full Intel module):

```bash
export LD_LIBRARY_PATH="/apps/spack/0.21/pitzer/linux-rhel9-skylake/intel-oneapi-compilers/gcc/11.4.1/2023.2.3-xq4aqvz/compiler/2023.2.3/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}"
```

### Files Modified
- `submit_snake_ppo.slurm` - Added Intel runtime library path
- `test_integration.slurm` - Added Intel runtime library path

---

## Issue #007: GLIBCXX Version Mismatch

**Job ID:** 43914204, 43914409
**Status:** RESOLVED
**Date:** 2025-01-19

### Error Message
```
ImportError: /apps/python/3.12/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by penv.so)
```

### Root Cause
PyTorch JIT compiles C++ extensions with GCC 13.2.0, which produces code requiring `GLIBCXX_3.4.32`. However, the system Python bundles an older `libstdc++.so.6` that doesn't have this symbol version.

Python's RPATH takes precedence over LD_LIBRARY_PATH, so simply adding the GCC lib path doesn't work.

### Solution
Use `LD_PRELOAD` to force loading the GCC 13.2.0 libstdc++:

```bash
export LD_LIBRARY_PATH="/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/apps/spack/0.21/pitzer/linux-rhel9-skylake/gcc/gcc/11.4.1/13.2.0-dveccoq/lib64/libstdc++.so.6"
```

Also clear cached torch extensions to force recompilation:
```bash
rm -rf /users/PAS3272/qifanwen/.cache/torch_extensions/py312_cu118/penv 2>/dev/null || true
```

### Files Modified
- `submit_snake_ppo.slurm` - Added LD_PRELOAD and cache clearing
- `test_integration.slurm` - Added LD_PRELOAD and cache clearing

---

## Issue #008: CUBLAS Deterministic Mode

**Job ID:** 43914410
**Status:** RESOLVED
**Date:** 2025-01-19

### Error Message
```
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)`
or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because
it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must
set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Root Cause
ALF enables deterministic mode for reproducibility. CUDA's cuBLAS library requires a specific workspace configuration to guarantee deterministic behavior.

### Solution
Set the CUBLAS workspace configuration environment variable:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Files Modified
- `submit_snake_ppo.slurm` - Added CUBLAS_WORKSPACE_CONFIG export
- `test_integration.slurm` - Added CUBLAS_WORKSPACE_CONFIG export

---

## Run History

| Job ID | Date | Status | Issue | Notes |
|--------|------|--------|-------|-------|
| 43912099 | 2025-01-19 | FAILED | #001 | Ninja PATH issue |
| 43914194 | 2025-01-19 | FAILED | #006 | libimf.so missing after GCC switch |
| 43914204 | 2025-01-19 | FAILED | #007 | GLIBCXX_3.4.32 not found |
| 43914409 | 2025-01-19 | FAILED | #007 | LD_LIBRARY_PATH insufficient, need LD_PRELOAD |
| 43914410 | 2025-01-19 | FAILED | #008 | CUBLAS deterministic mode config missing |
| 43914413 | 2025-01-19 | PASSED | - | Integration test: All 7 tests + 100 iterations |
| 43914415 | 2025-01-19 | RUNNING | - | Full training job submitted |

---

## Error Resolution Workflow

```
Job Failed?
    │
    ├─→ Check stderr: slurm_logs/snake_ppo_<job_id>.err
    │
    ├─→ Identify error pattern (see Quick Reference table)
    │
    ├─→ Apply fix from corresponding Issue section
    │
    ├─→ Run integration test: sbatch test_integration.slurm
    │
    └─→ If test passes, resubmit: sbatch submit_snake_ppo.slurm
```

## Adding New Issues

When encountering a new error:

1. Create a new section with format:
   ```markdown
   ## Issue #XXX: Brief Description

   **Job ID:** XXXXXXX
   **Status:** INVESTIGATING | RESOLVED | NOT A BUG
   **Date:** YYYY-MM-DD

   ### Error Message
   [Exact error text]

   ### Root Cause
   [Analysis]

   ### Solution
   [Fix steps]

   ### Files Modified
   [List of changes]
   ```

2. Add entry to Quick Reference table
3. Update Run History table
