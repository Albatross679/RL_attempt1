# RL Development Environment

RL environment combining DisMech (elastic rod simulator) with ALF (Agent Learning Framework).

## Setup

```bash
./setup.sh
```

This creates a virtual environment and installs all dependencies (PyTorch with CUDA, ALF, DisMech).

## Components

- **ALF**: Agent Learning Framework from Horizon Robotics
- **dismech-python**: Python bindings for DisMech discrete elastic rods simulator
- **dismech-rl**: RL environments using DisMech

## Verification

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import alf; print('ALF OK')"
python -c "import dismech; print('DisMech OK')"
```
