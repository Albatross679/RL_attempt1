# SLURM Usage Guide for RL Training

Guide for running reinforcement learning training jobs on OSC (Ohio Supercomputer Center).

## Quick Start

```bash
# 1. First-time setup (run once)
./setup.sh

# 2. Create log directory
mkdir -p slurm_logs

# 3. Run integration test (verify environment)
sbatch test_integration.slurm

# 4. Check test results
squeue -u $USER                                    # Wait for completion
cat slurm_logs/test_integration_*.out | tail -20   # Check results

# 5. Submit full training
sbatch submit_snake_ppo.slurm

# 6. Monitor training
./monitor_job.sh
```

---

## Initial Setup

### Prerequisites
- OSC account with GPU allocation
- Project account (e.g., PAS3272)

### Environment Setup

Run the setup script once to create the virtual environment:

```bash
cd /users/PAS3272/qifanwen/RL_attempt1
chmod +x setup.sh
./setup.sh
```

This installs:
- PyTorch with CUDA 11.8
- ALF reinforcement learning framework
- dismech physics simulation
- ninja build system (required for PyTorch JIT)

### Directory Structure

```
RL_attempt1/
├── .venv/                    # Python virtual environment
├── alf/                      # ALF framework (editable install)
├── dismech-python/           # Physics simulation (editable install)
├── dismech-rl/
│   ├── confs/               # Training configurations
│   └── scripts/             # Training scripts
├── results/                  # Training outputs
├── slurm_logs/              # SLURM job logs
├── submit_snake_ppo.slurm   # Main training job
├── test_integration.slurm   # Quick test job
├── monitor_job.sh           # Job monitor script
└── setup.sh                 # Environment setup
```

---

## Running on OSC (SLURM)

### Submit a Job

```bash
# Submit full training (4 hours, 256 parallel envs)
sbatch submit_snake_ppo.slurm

# Submit quick test (15 minutes, 32 envs)
sbatch test_integration.slurm
```

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>

# View job details
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# View all jobs with more details
squeue -u $USER -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

### View Output Logs

```bash
# List all log files
ls -la slurm_logs/

# View output (stdout)
cat slurm_logs/snake_ppo_<job_id>.out

# View errors (stderr)
cat slurm_logs/snake_ppo_<job_id>.err

# Follow output in real-time (while job is running)
tail -f slurm_logs/snake_ppo_<job_id>.out

# View last 50 lines
tail -50 slurm_logs/snake_ppo_<job_id>.out
```

### Cancel a Job

```bash
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Check GPU Allocation

```bash
# View available GPU partitions
sinfo -p gpu

# Check your remaining allocation
sacctmgr show associations user=$USER format=account%20,qos%20
```

---

## Monitoring Training

### Using monitor_job.sh

```bash
# Monitor most recent job
./monitor_job.sh

# Monitor specific job
./monitor_job.sh <job_id>

# Continuous monitoring (refreshes every 30s)
watch -n 30 ./monitor_job.sh
```

### Using TensorBoard

```bash
# Start TensorBoard (in a separate terminal or tmux)
source .venv/bin/activate
tensorboard --logdir results --port 6006

# If running remotely, use SSH tunnel:
# On your local machine:
ssh -L 6006:localhost:6006 user@owens.osc.edu
# Then open: http://localhost:6006
```

### Key Metrics to Watch

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `AverageReturn` | Mean episode reward | Should increase |
| `AverageEpisodeLength` | Steps per episode | Task-dependent |
| `loss/actor_loss` | Policy loss | Should decrease |
| `loss/critic_loss` | Value function loss | Should stabilize |
| `entropy` | Policy entropy | Should decrease slowly |

---

## Troubleshooting

### Quick Fixes

| Problem | Solution |
|---------|----------|
| "Ninja is required" error | Check that PATH export is before venv activation |
| Job stuck in PENDING | Check partition availability: `sinfo -p gpu` |
| CUDA out of memory | Reduce NUM_PARALLEL_ENVS to 128 or 64 |
| Module not found | Re-run `setup.sh` or check venv activation |

### Common Issues

See [SLURM_Issue_Doc.md](./SLURM_Issue_Doc.md) for detailed troubleshooting.

### Debug Mode

Run a quick integration test to verify the environment:

```bash
sbatch test_integration.slurm
# Wait for completion
cat slurm_logs/test_integration_*.out | grep -E "PASSED|FAILED"
```

---

## Customizing Training

### Environment Variables

Set these before submitting to customize training:

```bash
# Example: Custom training run
export NUM_PARALLEL_ENVS=128
export NUM_ITERATIONS=10000
export LEARNING_RATE=5e-4
sbatch submit_snake_ppo.slurm
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_PARALLEL_ENVS` | 256 | Parallel simulation environments |
| `NUM_ITERATIONS` | 5000 | Training iterations |
| `LEARNING_RATE` | 1e-3 | Actor/critic learning rate |
| `MINI_BATCH_SIZE` | 1024 | SGD mini-batch size |
| `UNROLL_LENGTH` | 4 | Steps before each update |
| `ENTROPY_REG` | 0.05 | Entropy regularization |
| `RENDER` | False | Enable visualization |

### Resume Training

```bash
# Resume from specific checkpoint
export ROOT_DIR=/path/to/previous/results/snake_approach_ppo_YYYYMMDD_HHMMSS
sbatch submit_snake_ppo.slurm
```

### Modify SLURM Resources

Edit `submit_snake_ppo.slurm`:

```bash
#SBATCH --time=08:00:00        # Increase time limit
#SBATCH --cpus-per-task=16     # More CPU cores
#SBATCH --gpus-per-node=2      # Multiple GPUs (if supported)
```

---

## Reference

### SLURM Commands Cheat Sheet

```bash
sbatch <script>        # Submit job
squeue -u $USER        # View your jobs
scancel <job_id>       # Cancel job
sacct -j <job_id>      # Job accounting info
sinfo -p gpu           # Partition info
scontrol show job <id> # Detailed job info
```

### File Locations

| File | Purpose |
|------|---------|
| `submit_snake_ppo.slurm` | Main training job script |
| `test_integration.slurm` | Quick environment test |
| `slurm_logs/*.out` | Job stdout logs |
| `slurm_logs/*.err` | Job stderr logs |
| `results/*/` | Training outputs, checkpoints |
| `results/*/train/` | TensorBoard logs |

### Useful Aliases

Add to your `~/.bashrc`:

```bash
alias sq='squeue -u $USER'
alias slogs='ls -lt slurm_logs/*.out | head -5'
alias slatest='cat $(ls -t slurm_logs/*.out | head -1)'
alias serr='cat $(ls -t slurm_logs/*.err | head -1)'
```
