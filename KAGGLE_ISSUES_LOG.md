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

---

## Run History

| Date | Kernel | Status | Duration | Notes |
|------|--------|--------|----------|-------|
| 2026-01-18 | snake_approach v1 | ERROR | ~1 min | llvmlite not found on PyTorch index |
