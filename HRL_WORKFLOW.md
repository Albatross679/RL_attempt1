# Hierarchical RL Workflow for Snake Predation

## Overview

This document describes the Hierarchical Reinforcement Learning (HRL) system for the snake predation task, where a snake robot must approach and coil around a cylindrical target.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Manager Agent (PPO)                │
│  State: [distance, wrap_angle, skill_id]        │
│  Action: skill selection (0=approach, 1=coil)   │
└──────────────────┬──────────────────────────────┘
                   │ selects skill
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐        ┌──────────────┐
│   Worker 1   │        │   Worker 2   │
│  (Approach)  │───────▶│   (Coil)     │
│   PPO        │ state  │   PPO        │
│              │ xfer   │              │
└──────────────┘        └──────────────┘
```

## Components

### Manager Agent (`SnakeHRLEnv`)

**Location:** `dismech-rl/environments/snake_hrl_env.py`

**State Space (Abstract):**
| Feature | Dim | Description |
|---------|-----|-------------|
| `distance_to_cylinder` | 1 | Head-to-cylinder distance |
| `wrap_angle` | 1 | Cumulative wrap angle |
| `current_skill` | 2 | One-hot: [approach, coil] |
| `skill_progress` | 2 | [approach_success, coil_success] |

**Action Space:** Discrete(2) - `{0: approach, 1: coil}`

**Reward:**
```python
reward = worker_reward
if skill_switched_successfully:
    reward += switch_bonus  # default: 1.0
if task_completed (full wrap):
    reward += completion_bonus  # default: 10.0
```

### Worker Agents

**Approach Worker** (`SnakeApproachEnv`)
- Learns undulation locomotion toward cylinder
- Success: head within `success_threshold` of cylinder
- Observes: node positions/velocities, cylinder relative position

**Coil Worker** (`SnakeCoilEnv`)
- Learns to wrap around cylinder
- Success: cumulative wrap angle > 2π
- Supports `set_initial_state()` for skill chaining

## Training Workflow

### Phase 1: Pre-train Workers (Optional but Recommended)

```bash
cd dismech-rl

# Train approach worker partially (~20k iterations)
NUM_ITERATIONS=20000 bash scripts/train_snake_approach_ppo.sh

# Train coil worker partially (~20k iterations)
NUM_ITERATIONS=20000 bash scripts/train_snake_coil_ppo.sh
```

Workers learn basic skills but don't need full convergence.

### Phase 2: Joint HRL Training

```bash
# Basic HRL training (workers use random actions)
bash scripts/train_snake_hrl_ppo.sh

# With pre-trained workers (recommended)
APPROACH_CHECKPOINT=results/snake_approach_ppo_*/ckpt-* \
COIL_CHECKPOINT=results/snake_coil_ppo_*/ckpt-* \
bash scripts/train_snake_hrl_ppo.sh
```

### Training Flow

1. Manager observes abstract state
2. Manager selects skill (approach=0 or coil=1)
3. Selected worker executes for N steps (default: 50)
4. If skill switch to coil:
   - Extract terminal state from approach worker
   - Initialize coil worker via `set_initial_state()`
5. Manager receives combined reward
6. Repeat until episode ends (coil success or timeout)

## Configuration

### Manager Config (`confs/snake_hrl_ppo_conf.py`)

Key parameters:
```python
worker_steps_per_manager_step = 50   # Worker steps per manager decision
switch_bonus = 1.0                    # Reward for successful skill switch
completion_bonus = 10.0               # Reward for full task completion
timeout_manager_steps = 100           # Episode timeout
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ITERATIONS` | 100000 | Training iterations |
| `NUM_PARALLEL_ENVS` | 64 | Parallel environment count |
| `WORKER_STEPS` | 50 | Worker steps per manager step |
| `SWITCH_BONUS` | 1.0 | Skill switch reward |
| `COMPLETION_BONUS` | 10.0 | Full wrap reward |

## State Transfer Mechanism

When switching from approach to coil:

```python
# In SnakeHRLEnv._switch_to_skill()
positions, velocities = approach_env.getVertices(), approach_env.getVelocities()
coil_env.set_initial_state(positions, velocities, cylinder)
```

This preserves:
- Snake node positions (continuity of shape)
- Node velocities (momentum)
- Cylinder target (same goal)

## Verification

```bash
# Run short training to verify setup
NUM_ITERATIONS=1000 bash scripts/train_snake_hrl_ppo.sh

# Monitor with TensorBoard
tensorboard --logdir results/snake_hrl_ppo_* --port 6006
```

Check for:
- Manager reward increasing
- Skill switches occurring (in env_info)
- State transfer working (coil starts from approach terminal)

## Debugging & Monitoring

### TensorBoard Metrics

The HRL environment tracks these metrics via `env_info`:

| Metric | Description |
|--------|-------------|
| `skill_selected` | Which skill was chosen (0 or 1) |
| `skill_0_selected` | Approach skill selection rate |
| `skill_1_selected` | Coil skill selection rate |
| `worker_reward` | Reward from worker execution |
| `switch_reward` | Bonus from successful skill switch |
| `distance_to_cylinder` | Current head-to-cylinder distance |
| `wrap_angle` | Current cumulative wrap angle |
| `approach_succeeded` | Approach success flag |
| `coil_succeeded` | Coil success flag |

View in TensorBoard:
```bash
tensorboard --logdir results/snake_hrl_ppo_* --port 6006
```

### Hyperparameter Tuning

Key parameters for tuning (defined in `confs/snake_hrl_ppo_conf.py`):

| Parameter | Default | Range | Rationale |
|-----------|---------|-------|-----------|
| `worker_steps` | 50 | 10, 25, 50, 100 | Balance exploration vs manager learning |
| `entropy_reg` | 0.1 | 0.01, 0.05, 0.1 | Control exploration vs exploitation |
| `switch_bonus` | 1.0 | 0.5, 1.0, 2.0 | Incentivize skill switching |
| `learning_rate` | 3e-4 | 1e-4, 3e-4, 1e-3 | Convergence speed |

Override via command line:
```bash
# Example: faster manager learning with less worker steps
python -m alf.bin.train --conf_file confs/snake_hrl_ppo_conf.py \
    --conf "worker_steps=25" --conf "entropy_reg=0.05"
```

## Known Issues & Solutions

### Issue: TypeError - NumPy Array Concatenation
**Symptom:** `TypeError: Concatenation operation is not implemented for NumPy arrays`
**Cause:** `worker.action_spec().sample()` returns ALF tensor, but worker expects numpy.
**Solution:** Convert action to numpy before stepping worker (fixed in snake_hrl_env.py:260-263).

### Issue: Workers Use Random Actions
**Current Status:** Workers execute random actions during HRL training.
**Solution:** Load pre-trained worker checkpoints (see Phase 1).
**Future:** Implement checkpoint loading in `SnakeHRLEnv.__init__()`.

### Issue: Skill Switch Timing
**Symptom:** Manager switches too early/late.
**Solution:** Adjust `switch_bonus` and ensure approach success threshold is appropriate.

### Issue: Exploration
**Symptom:** Manager gets stuck in one skill.
**Solution:** Increase `entropy_regularization` in PPO config.

## File Structure

```
dismech-rl/
├── environments/
│   ├── snake_hrl_env.py      # Manager environment (NEW)
│   ├── snake_approach_env.py # Approach worker
│   └── snake_coil_env.py     # Coil worker
├── confs/
│   ├── snake_hrl_ppo_conf.py # Manager config (NEW)
│   ├── snake_approach_ppo_conf.py
│   └── snake_coil_ppo_conf.py
└── scripts/
    ├── train_snake_hrl_ppo.sh # HRL training (NEW)
    ├── train_snake_approach_ppo.sh
    └── train_snake_coil_ppo.sh
```
