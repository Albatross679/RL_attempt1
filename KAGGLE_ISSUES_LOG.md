# Kaggle Issues Log

Track issues encountered during Kaggle training runs and their solutions.

## Template

```
### Issue: [Brief description]
**Date:** YYYY-MM-DD
**Kernel:** snake_approach / snake_coil
**Error:**
<error message>

**Solution:**
<what fixed it>
```

---

## Known Issues

### Issue: CUDA out of memory
**Symptoms:** `RuntimeError: CUDA out of memory`
**Solution:** Reduce batch size in config file or reduce network size.

### Issue: Git clone fails
**Symptoms:** `fatal: repository not found`
**Solution:** Ensure GitHub repo is public and URL is correct.

### Issue: Module not found after pip install
**Symptoms:** `ModuleNotFoundError: No module named 'dismech'`
**Solution:** Check that `pip install -e dismech-python/` completed successfully. May need to restart kernel.

### Issue: Submodule not initialized
**Symptoms:** Empty directories for alf/ or dismech-python/
**Solution:** Use `git clone --recursive` or run `git submodule update --init --recursive`

### Issue: llvmlite/triton not found on PyTorch index
**Symptoms:** `ERROR: Could not find a version that satisfies the requirement llvmlite==0.46.0`
**Solution:** Install llvmlite and triton from PyPI separately, not from PyTorch wheel index:
```bash
# First: Install PyTorch packages from PyTorch index
pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu118
# Then: Install llvmlite and triton from PyPI
pip install llvmlite==0.46.0 triton==3.3.1
```

### Issue: Submodules empty after clone (local commits not pushed)
**Symptoms:** `ERROR: dismech-python does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.`
**Root cause:** Git submodules are stored as references (gitlinks) to specific commits. When you make changes to a submodule and commit in the parent repo, the parent only stores a pointer to the commit hash. When someone clones with `--recursive`, Git tries to fetch those commits from the submodule's remote origin - but if your commits were never pushed there, the submodule ends up empty or at the wrong commit.
**Solution:** Either:
1. Fork each submodule repo, update remote URLs, push changes to forks, OR
2. Convert submodules to regular directories (embed the code directly)

See solution implementation below.

### Issue: torchvision/torchaudio +cu118 versions not found on PyPI
**Symptoms:** `ERROR: Could not find a version that satisfies the requirement torchvision==0.22.1+cu118`
**Root cause:** The alf package requires `torchvision==0.22.1+cu118` which is only available on the PyTorch wheel index, not PyPI. When pip installs alf's dependencies, it looks on PyPI and can't find this version.
**Solution:** Install the full PyTorch stack (torch, torchvision, torchaudio) from the PyTorch index BEFORE installing alf:
```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -e alf/
```

---

## Run History

| Date | Kernel | Status | Duration | Notes |
|------|--------|--------|----------|-------|
| 2026-01-18 | snake_approach v1 | ERROR | ~1 min | llvmlite not found on PyTorch index |
| 2026-01-18 | snake_approach v2 | ERROR | ~3 min | Submodule dismech-python empty (local commits not pushed) |
| 2026-01-18 | snake_approach v3 | ERROR | ~3 min | torchvision+cu118 not found on PyPI |
