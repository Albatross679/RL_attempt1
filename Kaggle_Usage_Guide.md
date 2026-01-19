# Kaggle Usage Guide

Instructions for configuring and using Kaggle with the DisMech RL training project.

---

## Prerequisites

- **Python 3.12** with virtual environment at `.venv312`
- **Kaggle account** at [kaggle.com](https://www.kaggle.com)
- **NVIDIA GPU** with CUDA support (optional, for GPU training)

---

## 1. Kaggle Configuration & Authentication

### Obtain API Token

1. Log in to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Click **"Create New Token"**
4. Copy the `KGAT_*` format token

### Set Environment Variable

Add to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export KAGGLE_API_TOKEN=KGAT_your_token_here
```

Or set per-session:
```bash
export KAGGLE_API_TOKEN=KGAT_35e34cc90119b3c30da316462e81b37c
```

### Verify Authentication

```bash
source .venv312/bin/activate
kaggle competitions list --page 1
```

If successful, you'll see a list of active Kaggle competitions.

---

## 2. Environment Setup

### Activate the Virtual Environment

```bash
source .venv312/bin/activate
```

### Verify Kaggle Installation

```bash
kaggle --version
# Expected output: Kaggle API 1.8.3 (or higher)
```

If not installed or outdated:
```bash
pip install --upgrade 'kaggle>=1.8'
```

---

## 3. Running Training Scripts

All training scripts are located in `./dismech-rl/scripts/`.

### Available Training Scripts

| Script | Algorithm | Task |
|--------|-----------|------|
| `train_follow_sac.sh` | SAC | Follow target (300 envs) |
| `train_follow_ppo.sh` | PPO | Follow target (256 envs) |
| `train_snake_approach_ppo.sh` | PPO | Snake approach (256 envs) |
| `train_snake_coil_ppo.sh` | PPO | Snake coiling (256 envs) |

### Basic Usage

```bash
cd dismech-rl/scripts
bash train_follow_sac.sh
```

The scripts automatically:
- Activate the `.venv312` environment
- Set up CUDA and threading optimizations
- Create timestamped results directories

### Custom Parameters

Override defaults with environment variables:

```bash
# Fewer iterations for testing
NUM_ITERATIONS=100 bash train_follow_sac.sh

# Custom output directory
ROOT_DIR=./my_results bash train_follow_sac.sh

# Multiple parameters
NUM_PARALLEL_ENVS=50 NUM_ITERATIONS=500 bash train_follow_ppo.sh
```

### Background Training with Logging

```bash
nohup bash train_follow_sac.sh > train.log 2>&1 &
tail -f train.log  # Monitor progress
```

### Resume from Checkpoint

```bash
ROOT_DIR=/path/to/previous/results bash train_follow_sac.sh
```

---

## 4. Monitoring Training

### TensorBoard

```bash
tensorboard --logdir dismech-rl/results --port 6006
```

Then open: http://localhost:6006

### Check GPU Usage

```bash
nvidia-smi -l 1  # Updates every second
```

---

## 5. Kaggle Datasets

### Download a Dataset

```bash
kaggle datasets download -d <dataset-owner>/<dataset-name>
unzip <dataset-name>.zip -d ./data/
```

### Download Competition Data

```bash
kaggle competitions download -c <competition-name>
```

---

## Quick Reference

```bash
# Full workflow example
export KAGGLE_API_TOKEN=KGAT_your_token_here
source .venv312/bin/activate
cd dismech-rl/scripts
bash train_follow_sac.sh
```

| Command | Description |
|---------|-------------|
| `kaggle --version` | Check Kaggle CLI version |
| `kaggle config view` | Show current configuration |
| `kaggle competitions list` | List active competitions |
| `kaggle datasets list -s <query>` | Search datasets |
