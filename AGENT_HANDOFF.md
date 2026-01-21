# AI Agent Handoff: Snake Approach RL Training

## Current State (2026-01-20 10:35 EST)

### Active Job
- **Job ID**: 43919533
- **Node**: p0253.ten.osc.edu
- **Status**: Running (~340/5000 iterations, ~7% complete)
- **Check status**: `squeue -u qifanwen`
- **View logs**: `tail -f /users/PAS3272/qifanwen/RL_attempt1/slurm_logs/snake_ppo_43919533.err`
- **Expected completion**: ~1 hour remaining

### Training Run Info (Run 2)
- **Run directory**: `/users/PAS3272/qifanwen/RL_attempt1/dismech-rl/results/snake_approach_ppo_20260120_101819`
- **Config**: `potential_type="simple_distance"` (PBRS reward shaping enabled)
- **Target**: Train snake to approach cylinder within 0.15m

### Previous Run Summary (Run 1 - Baseline)
| Metric | Value |
|--------|-------|
| Final return | -459.02 (at step 5000) |
| Improvement | +240 from initial -699 |
| Target reached | **NO** (all episodes timeout at 501) |
| Conclusion | Baseline insufficient → enabled PBRS |

### Current Run Metrics (Run 2 - PBRS)
| Metric | Value |
|--------|-------|
| Best return | -696.79 (early, step 150) |
| Latest return | -702.39 (step 300) |
| Target reached | Not yet (too early) |
| Trend | Initializing |

---

## Immediate Tasks for Next Agent

### 1. Monitor Current Job Progress
```bash
# Check if still running
squeue -j 43919533

# View recent logs
tail -20 /users/PAS3272/qifanwen/RL_attempt1/slurm_logs/snake_ppo_43919533.err
```

### 2. Check Training Metrics (after 1000+ steps)
```bash
cd /users/PAS3272/qifanwen/RL_attempt1
source .venv/bin/activate
python3 << 'EOF'
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('dismech-rl/results/snake_approach_ppo_20260120_101819/train')
ea.Reload()
returns = ea.Scalars('Metrics/AverageReturn')
ep_lengths = ea.Scalars('Metrics/AverageEpisodeLength')
for s in returns[-15:]:
    print(f"Step {s.step}: Return={s.value:.2f}")
# Check if any episodes < 501 (reached target)
success = [s for s in ep_lengths if 0 < s.value < 501]
print(f"\nEpisodes reaching target: {len(success)}")
EOF
```

### 3. If PBRS Run Shows Improvement
- Compare return curve with baseline Run 1
- Look for episodes with length < 501 (reaching target before timeout)
- Document findings in training_log.md

### 4. If PBRS Still Not Reaching Target
Try alternative potential functions:

**Option A: exp_distance (sharper near target)**
```bash
# Edit snake_approach_ppo_conf.py line 33:
potential_type="exp_distance",
potential_params={'sigma': 0.3},
```

**Option B: distance_alignment (adds heading guidance)**
```bash
potential_type="distance_alignment",
potential_params={'align_weight': 0.5},
```

---

## Key Files

| File | Purpose |
|------|---------|
| `dismech-rl/confs/snake_approach_ppo_conf.py` | PPO config, change `potential_type` on line 33 |
| `dismech-rl/results/snake_approach_ppo_20260120_101819/` | Current PBRS training results |
| `dismech-rl/results/snake_approach_ppo_20260120_062711/` | Baseline training results |
| `training_log.md` | Training progress documentation |
| `submit_snake_ppo.slurm` | SLURM job submission script |

---

## Success Criteria

1. **AverageReturn** should improve faster than baseline
2. **AverageEpisodeLength** should decrease below 501 (snake reaching target before timeout)
3. Target is reached when snake gets within 0.15m of cylinder

---

## Context

The snake robot uses undulation locomotion (curvature control) to move. Run 1 (baseline, no PBRS) showed the snake learning but never reaching the target in 5000 iterations. Run 2 uses `simple_distance` PBRS to provide denser reward feedback as the snake approaches the target.

The environment:
- 76-dim observation (positions, velocities, curvatures, target, heading)
- 10-dim action (curvature control for 5 body segments)
- 500 step timeout, success at dist < 0.15m
