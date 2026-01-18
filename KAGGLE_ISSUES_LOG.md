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

---

## Run History

| Date | Kernel | Status | Duration | Notes |
|------|--------|--------|----------|-------|
| | | | | |
