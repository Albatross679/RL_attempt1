# Snake Approach Training Log

## Overview
Training a snake robot to approach a cylindrical target using PPO with reward shaping.

---

## Run 1: Baseline (potential_type="none")

### Configuration
| Parameter | Value |
|-----------|-------|
| Run ID | `snake_approach_ppo_20260120_062711` |
| Algorithm | PPO |
| Potential Type | `none` (baseline) |
| Total Iterations | 5000 |
| Parallel Envs | 256 |
| Hidden Layers | (256, 256, 256) |
| Learning Rate | 1e-3 |
| Entropy Reg | 0.05 |

### Reward Structure (Baseline)
```python
reward = -dist²                    # Distance penalty (primary)
       + 0.1 * max(0, fwd_vel)     # Forward velocity bonus
       + {2.0 if dist<0.1 else 0.5 if dist<0.2 else 0}  # Proximity bonuses
```

### Training Progress

| Step | AverageReturn | Status |
|------|--------------|--------|
| 150 | -699.43 | Initial |
| 500 | -740.17 | Exploring |
| 1000 | -718.45 | Learning |
| 1500 | -685.90 | Improving |
| 2000 | -685.56 | Plateau |
| 2500 | -634.09 | Improving |
| 3000 | -616.42 | Improving |
| 3150 | -545.31 | **Best so far** |

**Episode Length**: Constant at 501 (always timing out, never reaching target)

### Analysis
- **Trend**: Reward is **IMPROVING** (less negative is better)
- **Rate**: ~50 reward improvement per 1000 steps
- **Issue**: Snake always times out (500 steps) - not reaching target yet
- **Progress**: 3150/5000 iterations (63%)

### Current Job
- **Job ID**: 43919349
- **Node**: p0232.ten.osc.edu
- **TensorBoard Port**: 6349
- **Connect Command**: `ssh -N -L 6349:p0232.ten.osc.edu:6349 qifanwen@pitzer.osc.edu`

---

## Observations

### What's Working
1. Reward is steadily improving
2. Snake is learning to reduce distance penalty
3. Training throughput is stable (~1800-2000 samples/sec)

### Potential Issues
1. Never reaching target (always 501 episode length)
2. May need reward shaping if learning plateaus before success

---

## Next Steps (if reward stalls after 5000 iterations)

### Option A: Enable PBRS with `simple_distance`
```bash
# Edit confs/snake_approach_ppo_conf.py line 33:
potential_type="simple_distance",
potential_params={'scale': 1.0, 'd_max': 2.0},
```

### Option B: Try `exp_distance` for sharper reward near target
```bash
potential_type="exp_distance",
potential_params={'sigma': 0.3},
```

### Option C: Try `distance_alignment` for heading guidance
```bash
potential_type="distance_alignment",
potential_params={'align_weight': 0.5},
```

---

## Checkpoints Available
| Checkpoint | Step | Notes |
|------------|------|-------|
| ckpt-500 | 500 | Early exploration |
| ckpt-1000 | 1000 | Learning signal emerging |
| ckpt-1500 | 1500 | |
| ckpt-2000 | 2000 | |
| ckpt-2500 | 2500 | |
| ckpt-3000 | 3000 | Current best |

---

## Updates

### 2026-01-20 07:30
- Started training run with baseline configuration
- Training reached 3000 iterations, job timed out
- Resumed training from ckpt-3000
- Reward improving: -699 -> -545 over 3000 steps
- Still not reaching target (always timing out)

### 2026-01-20 09:03 (Run 1 COMPLETED)
**Final Training Status**: Job 43919349 COMPLETED (5000/5000 iterations)

**Final Metrics**:
| Milestone | AverageReturn |
|-----------|--------------|
| Step 500 | -740.17 |
| Step 1000 | -718.45 |
| Step 2000 | -685.56 |
| Step 3000 | -616.42 |
| Step 4000 | -575.06 |
| Step 4500 | -573.65 |
| Step 4800 | -496.06 |
| Step 5000 | **-459.02** (Final, Best) |

**Final Analysis**:
- Total improvement: +240 points (from -699 to -459)
- Episodes reaching target: **0** (all timeout at 501 steps)
- **Conclusion**: Baseline reward insufficient for target approach
- **Action**: Enabled PBRS with `simple_distance` for Run 2

---

## Run 2: PBRS with simple_distance (potential_type="simple_distance")

### Configuration
| Parameter | Value |
|-----------|-------|
| Run ID | `snake_approach_ppo_20260120_101819` |
| Algorithm | PPO |
| Potential Type | `simple_distance` |
| Total Iterations | 5000 |
| Job ID | 43919533 |
| Node | p0253 |

### PBRS Reward Shaping
The `simple_distance` potential function adds shaped reward:
```python
Φ(s) = (d_max - dist) / d_max  # Normalized distance-based potential
shaped_reward = γ * Φ(s') - Φ(s)  # Difference gives approach bonus
```

### Training Progress
| Step | AverageReturn | EpisodeLength | Status |
|------|--------------|---------------|--------|
| 0 | - | - | Started 10:18 EST |
| 150 | -696.79 | 501 | Initial metrics |
| 300 | -702.39 | 501 | Exploring |

### 2026-01-20 10:18 (Run 2 Started)
- Changed `potential_type` from "none" to "simple_distance"
- Submitted job 43919533, running on p0253
- Throughput: ~1990 samples/sec

### 2026-01-20 10:34 (Early Progress Update)
- Training at step ~340/5000 (~7%)
- Initial return: -696 to -702 (similar to baseline start)
- All episodes still timing out at 501 (expected at early stage)
- PBRS shaping benefits typically visible as snake gets closer to target
- Expected completion: ~1 hour remaining
