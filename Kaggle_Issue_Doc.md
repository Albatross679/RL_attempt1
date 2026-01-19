# Kaggle Environment Setup - Troubleshooting Log

This document records all issues encountered during Kaggle environment setup and script execution, along with their solutions.

---

## Issue List

### 1. Kaggle Package Version Too Old (System Python)

**Problem:** System Python 3.10 had Kaggle 1.7.4.5 installed, which is below the required minimum version (1.8+).

**Solution:** Installed Kaggle 1.8.3 in the `.venv312` Python 3.12 virtual environment:
```bash
source .venv312/bin/activate
pip install --upgrade 'kaggle>=1.8'
```
**Result:** ✅ Kaggle 1.8.3 successfully installed with all dependencies.

---

### 2. Kaggle Not Installed in Project Virtual Environment

**Problem:** The `.venv312` virtual environment used by the training scripts did not have the Kaggle package installed.

**Solution:** Installed the Kaggle package specifically in the project's virtual environment (same command as Issue #1).

**Result:** ✅ Kaggle available in `.venv312`.

---

### 3. Legacy API Token Authentication Failed (401 Unauthorized)

**Problem:** Initially configured Kaggle credentials in `~/.kaggle/kaggle.json` with a legacy-format API token. API calls returned:
```
401 Client Error: Unauthorized for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/ListCompetitions
```

**Solution:** Kaggle 1.8+ supports a new token format (`KGAT_*`) via environment variable. Instead of using `kaggle.json`, set the token as an environment variable:
```bash
export KAGGLE_API_TOKEN=KGAT_35e34cc90119b3c30da316462e81b37c
```

**Result:** ✅ Authentication successful - `kaggle competitions list` returns valid data.

---

### 4. All Training Scripts Execute Successfully

**Verification:** Tested all 4 training scripts:

| Script | Status | Notes |
|--------|--------|-------|
| `train_follow_sac.sh` | ✅ Pass | 300 parallel environments, SAC algorithm |
| `train_follow_ppo.sh` | ✅ Pass | 256 parallel environments, PPO algorithm |
| `train_snake_approach_ppo.sh` | ✅ Pass | 256 parallel environments, snake approach task |
| `train_snake_coil_ppo.sh` | ✅ Pass | 256 parallel environments, snake coil task |

All scripts:
- Auto-activate the `.venv312` environment
- Load PyTorch CUDA extensions correctly
- Spawn parallel training processes
- Initialize training infrastructure properly

**Result:** ✅ All training scripts are fully functional.

---

## Summary

| Issue | Status | Resolution |
|-------|--------|------------|
| Kaggle version < 1.8 | ✅ Resolved | Upgraded to 1.8.3 in venv312 |
| Kaggle not in venv | ✅ Resolved | pip install in venv312 |
| 401 Unauthorized (legacy token) | ✅ Resolved | Use KAGGLE_API_TOKEN env var |
| Training scripts | ✅ All Working | No issues found |

---

## Key Takeaways

1. **Token Format**: Kaggle 1.8+ uses `KGAT_*` prefixed tokens set via `KAGGLE_API_TOKEN` environment variable
2. **Virtual Environment**: Always install Kaggle in the project's virtual environment (`.venv312`)
3. **Scripts**: All training scripts work out-of-box with proper environment setup
