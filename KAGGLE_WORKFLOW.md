# Kaggle Training Workflow

## Overview

This workflow uses Kaggle's free GPU resources to train reinforcement learning models. The approach clones your public GitHub repository directly into Kaggle kernels.

## Prerequisites

1. **Public GitHub repo** with all code committed (including submodule changes)
2. **Kaggle account** with API credentials configured

## Initial Setup

### 1. Configure Kaggle API Credentials

```bash
# Create Kaggle config directory
mkdir -p ~/.kaggle

# Create kaggle.json with your API token
# Get token from: https://www.kaggle.com/settings -> API -> Create New Token
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"albatross679","key":"KGAT_e74f5ab1bbe761eea1ad5ffb4797e2d4"}
EOF

chmod 600 ~/.kaggle/kaggle.json
```

### 2. Verify Kaggle CLI

```bash
# Test authentication
kaggle competitions list

# Should see list of competitions without errors
```

## Usage

### Push Code to GitHub

Before each Kaggle run, ensure your latest code is pushed:

```bash
# Commit any submodule changes first
cd dismech-python && git add -A && git commit -m "updates" && cd ..
cd alf && git add -A && git commit -m "updates" && cd ..

# Update submodule references in main repo
git add dismech-python alf dismech-rl
git commit -m "Update submodules"
git push origin master
```

### Run Snake Approach Training

```bash
# Push kernel to Kaggle
kaggle kernels push -p kaggle/snake_approach/

# Check status
kaggle kernels status albatross679/snake-approach-ppo

# View logs (after completion)
kaggle kernels output albatross679/snake-approach-ppo -p ./kaggle_results/snake_approach/
```

### Run Snake Coil Training

```bash
# Push kernel
kaggle kernels push -p kaggle/snake_coil/

# Check status
kaggle kernels status albatross679/snake-coil-ppo

# Get output
kaggle kernels output albatross679/snake-coil-ppo -p ./kaggle_results/snake_coil/
```

## Kaggle Limits

| Resource | Limit |
|----------|-------|
| GPU | 30 hours/week |
| GPU session | 12 hours max |
| TPU | 20 hours/week |
| TPU session | 9 hours max |
| Internet | Required (enabled in kernels) |

## Troubleshooting

### Check Kernel Logs

```bash
# Get full output including logs
kaggle kernels output <username>/<kernel-name> -p ./output/
```

### Common Issues

1. **Kernel fails immediately**: Check that GPU is enabled in kernel-metadata.json
2. **Git clone fails**: Ensure repo is public
3. **Import errors**: Dependencies may have changed, update requirements-cuda.txt
4. **CUDA errors**: Kaggle uses specific CUDA versions, ensure compatibility

### View Running Kernels

Visit https://www.kaggle.com/code to see kernel status in web UI.

## Directory Structure

```
kaggle/
├── snake_approach/
│   ├── kaggle_train.py      # Training script
│   └── kernel-metadata.json  # Kaggle kernel config
└── snake_coil/
    ├── kaggle_train.py
    └── kernel-metadata.json
```

## Retrieving Results

After training completes:

```bash
# Download output files
kaggle kernels output albatross679/snake-approach-ppo -p ./results/

# Output includes:
# - Trained model checkpoints
# - TensorBoard logs
# - Training metrics
```

## Tips

1. **Monitor progress**: Use `kaggle kernels status` or web UI
2. **Save checkpoints**: Results in `/kaggle/working/` are preserved
3. **Log to issues**: Update KAGGLE_ISSUES_LOG.md with any problems encountered
4. **Iterate quickly**: Test locally first, then push to Kaggle for full training runs
