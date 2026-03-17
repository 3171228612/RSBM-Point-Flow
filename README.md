# RSBM-Point-Flow: Rectified Schrödinger Bridge Matching for Point Cloud Registration

Based on [Rectified Point Flow](https://github.com/GradientSpaces/Rectified-Point-Flow) (NeurIPS 2025 Spotlight).

**TL;DR:** Replace RPF's flow matching with ε-Rectified Schrödinger Bridge, achieving comparable accuracy with **5-10× fewer sampling steps**.

---

## Quick Start (H200 8-GPU)

### 1. Clone & Setup

```bash
cd /mlx_devbox/users/zhaoliangjie
git clone https://github.com/3171228612/RSBM-Point-Flow.git Rectified-Point-Flow-main
cd Rectified-Point-Flow-main
```

### 2. Install Dependencies

> **Prerequisites:** PyTorch 2.8.0 + CUDA 12.x already installed on the machine.
>
> ⚠️ **Do NOT install `torch`** — the script will use the existing one.
>
> ⚠️ **Do NOT install `xformers`** — it's not used and will break torch version.

```bash
bash install_h200.sh
```

### 3. Download Datasets (~86 GB total)

```bash
bash run_full_experiments.sh phase1
```

This downloads all 6 datasets (with correct filenames):

| # | Dataset | Size | Filename |
|---|---------|------|----------|
| 1 | IKEA | 293 MB | `ikea.hdf5` |
| 2 | PartNet | 52 GB | `partnet.hdf5` |
| 3 | BreakingBad-Everyday | 27 GB | ⚠️ `everyday.hdf5` (renamed from `breaking_bad_vol.hdf5`) |
| 4 | Two-by-Two | 259 MB | ⚠️ `twobytwo.hdf5` (renamed from `2by2.hdf5`) |
| 5 | ModelNet-40 | 2 GB | `modelnet.hdf5` |
| 6 | TUD-L | 4 GB | `tudl.hdf5` |

Pretrained RPF encoder checkpoint is auto-downloaded from HuggingFace.

### 4. Verify Multi-GPU (DDP Test)

```bash
bash run_full_experiments.sh verify
```

Quick test: 2 GPUs, 2 epochs, IKEA only. Takes ~5 minutes. If it passes, multi-GPU is working.

### 5. Train (4+4 GPUs, ~3-4 days)

```bash
# Run in background
nohup bash run_full_experiments.sh phase2 > train.log 2>&1 &

# Monitor progress
tail -f train.log
```

This trains **RSBM** (GPU 0-3) and **RPF baseline** (GPU 4-7) simultaneously, 2000 epochs each, 6 datasets.

### 6. Ablation Experiments (~30 min)

After training completes:

```bash
bash run_full_experiments.sh phase3
```

Runs full ablation matrix:
- **Methods:** RSBM vs RPF
- **Datasets:** ikea, partnet, everyday, twobytwo, modelnet, tudl
- **Schedules (RSBM):** linear, uniform, karras
- **Steps:** 3, 5, 10, 20, 30, 50

### 7. Speed Benchmark (~10 min)

```bash
bash run_full_experiments.sh phase4
```

---

## Manual Commands

### Train RSBM only

```bash
python train.py --config-name "RSBM_main" \
    data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
    trainer.devices=8 \
    trainer.strategy="ddp" \
    data.batch_size=40 \
    data.num_workers=16 \
    model.encoder_ckpt="./weights/RPF_base_full_anchorfree_ep2000.ckpt" \
    +model.frozen_encoder=true
```

### Train RPF baseline (same conditions)

```bash
python train.py --config-name "RPF_base_main" \
    data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
    trainer.devices=8 \
    trainer.strategy="ddp" \
    data.batch_size=40 \
    data.num_workers=16 \
    model.encoder_ckpt="./weights/RPF_base_full_anchorfree_ep2000.ckpt" \
    +model.frozen_encoder=true
```

### Run Inference

```bash
# RSBM with 5 steps
python sample.py --config-name RSBM_demo \
    ckpt_path="./output/RSBM_6ds_ep2000/last.ckpt" \
    data=ikea \
    data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
    model.inference_sampling_steps=5 \
    visualizer.renderer=none

# RPF with 50 steps
python sample.py --config-name RPF_base_demo \
    ckpt_path="./output/RPF_6ds_ep2000/last.ckpt" \
    data=ikea \
    data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
    model.inference_sampling_steps=50 \
    visualizer.renderer=none
```

---

## Project Structure (RSBM additions)

```
Rectified-Point-Flow-main/
├── rectified_point_flow/
│   ├── rsbm_modeling.py          # RSBM model (inherits RPF)
│   ├── rsbm_pcr_bridge.py        # ε-Rectified Bridge math
│   ├── rsbm_sampler.py           # Heun/Euler ODE samplers
│   ├── modeling.py               # Original RPF model
│   └── flow_model/layer.py       # Modified: flash_attn fallback + torch.compile fix
├── config/
│   ├── RSBM_main.yaml            # RSBM training config
│   ├── RSBM_demo.yaml            # RSBM inference config
│   └── model/rsbm_point_flow.yaml # RSBM model config (epsilon, schedule, etc.)
├── install_h200.sh               # H200 environment setup
├── run_full_experiments.sh        # Full experiment pipeline
├── run_schedule_ablation.sh       # Schedule ablation script
└── benchmark_inference.py         # Speed benchmark
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.epsilon` | 0.5 | Bridge stochasticity (higher = more stochastic) |
| `model.inference_sampling_steps` | 50 | Steps during validation (override at test time) |
| `model.inference_sampler` | heun | ODE solver: `heun` (2nd order) or `euler` (1st order) |
| `model.sigma_schedule` | linear | Sigma schedule: `linear`, `uniform`, `karras` |
| `model.timestep_sampling` | uniform | Training timestep distribution (must be `uniform` for RSBM) |

## Known Issues

- **`@torch.compile`**: Disabled in `layer.py` due to PyTorch 2.8 `CuteDSLBenchmarkRequest` bug.
- **`xformers`**: Do NOT install. Not used by this codebase and will overwrite torch.
- **Dataset filenames**: BreakingBad must be saved as `everyday.hdf5`, Two-by-Two as `twobytwo.hdf5`.

## Based On

- [Rectified Point Flow](https://github.com/GradientSpaces/Rectified-Point-Flow) — NeurIPS 2025 Spotlight
- [Rectified Schrödinger Bridge Matching](https://arxiv.org/abs/2401.XXXXX) — Bridge-based generative modeling
