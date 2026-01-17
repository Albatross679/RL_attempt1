# RL Development Environment

RL environment combining DisMech (elastic rod simulator) with ALF (Agent Learning Framework).

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (already done during setup)
pip install -e ./alf
pip install -e ./dismech-python
pip install pyvista pyelastica==0.2.4 tensorboard huggingface-hub
```

## Components

- **ALF**: Agent Learning Framework from Horizon Robotics
- **dismech-python**: Python bindings for DisMech discrete elastic rods simulator
- **dismech-rl**: RL environments using DisMech

## Verification

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import alf; print('ALF OK')"
python -c "import dismech; print('DisMech OK')"
```

## Notes

- PyTorch installed with CUDA 11.8 support
- ALF modified to use CUDA 11.8 (originally cu128)
- dismech-python modified to support Python 3.10 (originally required 3.13)
- atari_py removed from ALF deps (optional, requires cmake)
