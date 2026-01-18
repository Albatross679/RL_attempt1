# Deploying RL_attempt1 to a Remote Linux Server

## TL;DR - Can You Just Clone and Run?

**Almost.** You need to:
1. Push to GitHub (with submodules configured properly)
2. Clone on server with `--recursive`
3. Recreate `.venv` (can't copy - has hardcoded paths)
4. Install dependencies
5. Run training

---

## Step 1: Push to GitHub

The repo already has proper submodules configured. Just push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/RL_attempt1.git
git push -u origin master
```

---

## Step 2: Clone on Remote Server

### For OSC (Ohio Supercomputer Center)
```bash
ssh username@owens.osc.edu

module load python/3.10
module load cuda/11.8

cd $HOME
git clone --recursive https://github.com/YOUR_USERNAME/RL_attempt1.git
cd RL_attempt1
```

### For AWS EC2
```bash
ssh -i your-key.pem ubuntu@<ec2-ip>

sudo apt update
sudo apt install python3.10 python3.10-venv

cd ~
git clone --recursive https://github.com/YOUR_USERNAME/RL_attempt1.git
cd RL_attempt1
```

---

## Step 3: Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -e ./alf
pip install -e ./dismech-python
pip install -e .
```

---

## Step 4: Run Training

```bash
cd RL_attempt1/dismech-rl/scripts
source ../../.venv/bin/activate

# Run PPO training
bash train_follow_ppo.sh

# Or with custom parameters
NUM_ITERATIONS=5000 NUM_PARALLEL_ENVS=128 bash train_follow_ppo.sh
```

---

## Step 5: Monitor Training

### TensorBoard
```bash
# On server (in another terminal/tmux)
tensorboard --logdir ~/RL_attempt1/dismech-rl/results --port 6006 --bind_all

# SSH tunnel from local machine
ssh -L 6006:localhost:6006 username@server
# Then open http://localhost:6006
```

### For OSC (SLURM Job)
Create `train_job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=YOUR_ACCOUNT

module load python/3.10 cuda/11.8
source ~/RL_attempt1/.venv/bin/activate
cd ~/RL_attempt1/dismech-rl/scripts
bash train_follow_ppo.sh
```
Submit: `sbatch train_job.sh`

---

## What NOT to Copy

| Item | Copy? | Reason |
|------|-------|--------|
| `.venv/` | NO | Has hardcoded paths, recreate on server |
| `results/` | NO | Training outputs, generate fresh |
| `__pycache__/` | NO | Regenerated automatically |

---

## Troubleshooting

### CUDA Version Mismatch
**Symptom:** `CUDA error: no kernel image is available`
**Fix:** Check server CUDA version with `nvidia-smi`, then install matching PyTorch:
```bash
# For CUDA 12.1
pip install torch==2.7.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
**Symptom:** OOM during training
**Fix:** Reduce parallel environments:
```bash
NUM_PARALLEL_ENVS=64 bash train_follow_ppo.sh
```

### Permission Denied on Scripts
**Fix:** `chmod +x dismech-rl/scripts/*.sh`

### Module Not Found
**Fix:** Ensure editable installs:
```bash
pip install -e ./alf -e ./dismech-python -e .
```

---

## Quick Checklist

- [ ] Push repo to GitHub
- [ ] Clone on server with `--recursive`
- [ ] Create fresh `.venv` with Python 3.10
- [ ] Install PyTorch with correct CUDA version
- [ ] `pip install -e` for alf, dismech-python, root
- [ ] `chmod +x` scripts
- [ ] Test: `NUM_ITERATIONS=100 bash train_follow_ppo.sh`
