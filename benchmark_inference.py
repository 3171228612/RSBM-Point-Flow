"""
Benchmark inference speed: RSBM vs RPF.

Usage:
    # Benchmark RSBM (3 Heun steps, NFE=5)
    python benchmark_inference.py --config-name RSBM_demo \
        data=small4 \
        data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
        ckpt_path="./output/RSBM_eps05_heun3/last.ckpt"

    # Benchmark RPF baseline (50 Euler steps, NFE=50)
    python benchmark_inference.py --config-name RPF_base_demo \
        data=small4 \
        data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
        ckpt_path="./weights/RPF_base_full_anchorfree_ep2000.ckpt"

    # Quick comparison with fewer warmup/runs
    python benchmark_inference.py --config-name RSBM_demo \
        data=small4 \
        data_root="/mlx_devbox/users/zhaoliangjie/dataset" \
        ckpt_path="./output/RSBM_eps05_heun3/last.ckpt" \
        +benchmark.num_warmup=5 \
        +benchmark.num_runs=20
"""

import logging
import os
import time
import warnings
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module, download_rfp_checkpoint

logger = logging.getLogger("Benchmark")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_CKPT_PATH_HF = "RPF_base_full_anchorfree_ep2000.ckpt"


@hydra.main(version_base="1.3", config_path="./config", config_name="RSBM_demo")
def main(cfg: DictConfig):
    """Benchmark inference speed."""

    # ---- Setup ----
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        ckpt_path = download_rfp_checkpoint(DEFAULT_CKPT_PATH_HF, './weights')
    elif not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        exit(1)

    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True, verbose=False)

    # Build model & data
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()
    model.cuda()

    # Prepare data
    datamodule.setup("test")
    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, list):
        val_loader = val_loader[0]

    # Get benchmark config
    benchmark_cfg = cfg.get("benchmark", {})
    num_warmup = benchmark_cfg.get("num_warmup", 10)
    num_runs = benchmark_cfg.get("num_runs", 50)

    # ---- Get a batch of real data ----
    data_dict = next(iter(val_loader))
    # Move to GPU
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()

    batch_size = data_dict["pointclouds"].shape[0]
    num_points = data_dict["pointclouds"].shape[1]

    # Model info
    sampler_name = getattr(model, 'inference_sampler', 'euler')
    num_steps = getattr(model, 'inference_sampling_steps', 50)
    epsilon = getattr(model, 'epsilon', 'N/A')
    model_name = model.__class__.__name__

    print("\n" + "=" * 70)
    print(f"  INFERENCE SPEED BENCHMARK")
    print("=" * 70)
    print(f"  Model:          {model_name}")
    print(f"  Sampler:        {sampler_name}")
    print(f"  Steps:          {num_steps}")
    print(f"  Epsilon:        {epsilon}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Num points:     {num_points}")
    print(f"  Warmup runs:    {num_warmup}")
    print(f"  Benchmark runs: {num_runs}")
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ---- Warmup ----
    print(f"\n[1/3] Warming up ({num_warmup} runs)...")
    with torch.inference_mode():
        for _ in range(num_warmup):
            latent = model._encode(data_dict)
            _ = model.sample_rectified_flow(data_dict, latent)
    torch.cuda.synchronize()

    # ---- Benchmark: Full pipeline (encode + sample) ----
    print(f"[2/3] Benchmarking full pipeline ({num_runs} runs)...")
    torch.cuda.synchronize()

    # Use CUDA events for accurate GPU timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.inference_mode():
        for i in range(num_runs):
            start_events[i].record()
            latent = model._encode(data_dict)
            result = model.sample_rectified_flow(data_dict, latent)
            end_events[i].record()

    torch.cuda.synchronize()
    full_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # ---- Benchmark: Sampling only (no encoding) ----
    print(f"[3/3] Benchmarking sampling only ({num_runs} runs)...")
    with torch.inference_mode():
        latent = model._encode(data_dict)  # Pre-compute features

    start_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    with torch.inference_mode():
        for i in range(num_runs):
            start_events2[i].record()
            result = model.sample_rectified_flow(data_dict, latent)
            end_events2[i].record()

    torch.cuda.synchronize()
    sample_times = [s.elapsed_time(e) for s, e in zip(start_events2, end_events2)]

    # ---- Compute NFE ----
    if sampler_name == "heun":
        nfe = 2 * num_steps - 1
    elif sampler_name in ("rk2",):
        nfe = 2 * num_steps
    elif sampler_name in ("rk4",):
        nfe = 4 * num_steps
    else:  # euler
        nfe = num_steps

    # ---- Results ----
    avg_full = sum(full_times) / len(full_times)
    std_full = (sum((t - avg_full) ** 2 for t in full_times) / len(full_times)) ** 0.5
    min_full = min(full_times)
    max_full = max(full_times)

    avg_sample = sum(sample_times) / len(sample_times)
    std_sample = (sum((t - avg_sample) ** 2 for t in sample_times) / len(sample_times)) ** 0.5
    min_sample = min(sample_times)
    max_sample = max(sample_times)

    print("\n" + "=" * 70)
    print(f"  RESULTS: {model_name} ({sampler_name}, {num_steps} steps)")
    print("=" * 70)
    print(f"\n  Full Pipeline (Encode + Sample):")
    print(f"    Mean:  {avg_full:.2f} ms  (+/- {std_full:.2f} ms)")
    print(f"    Min:   {min_full:.2f} ms")
    print(f"    Max:   {max_full:.2f} ms")
    print(f"\n  Sampling Only (flow ODE integration):")
    print(f"    Mean:  {avg_sample:.2f} ms  (+/- {std_sample:.2f} ms)")
    print(f"    Min:   {min_sample:.2f} ms")
    print(f"    Max:   {max_sample:.2f} ms")
    print(f"\n  NFE (Number of Function Evaluations): {nfe}")
    print(f"  Throughput: {1000.0 / avg_full:.1f} samples/sec (full pipeline)")
    print(f"  Throughput: {1000.0 / avg_sample:.1f} samples/sec (sampling only)")
    print("=" * 70)

    # ---- Save results to file ----
    out_dir = Path(cfg.get("log_dir", "./output/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"benchmark_{model_name}_{sampler_name}_{num_steps}steps.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Sampler: {sampler_name}\n")
        f.write(f"Steps: {num_steps}\n")
        f.write(f"NFE: {nfe}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Num points: {num_points}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Num warmup: {num_warmup}\n")
        f.write(f"Num runs: {num_runs}\n")
        f.write(f"\nFull Pipeline (ms):\n")
        f.write(f"  Mean: {avg_full:.2f}\n")
        f.write(f"  Std:  {std_full:.2f}\n")
        f.write(f"  Min:  {min_full:.2f}\n")
        f.write(f"  Max:  {max_full:.2f}\n")
        f.write(f"\nSampling Only (ms):\n")
        f.write(f"  Mean: {avg_sample:.2f}\n")
        f.write(f"  Std:  {std_sample:.2f}\n")
        f.write(f"  Min:  {min_sample:.2f}\n")
        f.write(f"  Max:  {max_sample:.2f}\n")
        f.write(f"\nAll full pipeline times (ms): {full_times}\n")
        f.write(f"All sampling times (ms): {sample_times}\n")

    print(f"\n  Results saved to: {result_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
