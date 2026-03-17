"""
Rectified Schrödinger Bridge Matching (RSBM) for Point Cloud Registration.

Inherits from RectifiedPointFlow (RPF) and replaces the core flow matching logic:
  - RPF:  linear interpolation x_t = (1-t)x_0 + tx_1,  constant velocity v_t = x_1 - x_0
  - RSBM: ε-rectified bridge  x_t = μ_t + σ_{ε,t}·noise,  time-varying velocity target

Key advantages over RPF:
  1. ε-rectification: controllable path straightness → fewer sampling steps
  2. Time-varying velocity target: richer learning signal than constant velocity
  3. Heun (2nd-order) sampler: NFE=5 for 3 steps vs RPF's NFE=50 for 50 Euler steps
  4. ε-invariant velocity target: same network works across different ε schedules

All RPF infrastructure is preserved: PTv3 encoder, DiT flow model, evaluator, data pipeline.

Reference: "Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation"
"""

import math
from functools import partial
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling import RectifiedPointFlow
from .rsbm_pcr_bridge import RSBMPointCloudBridgeSimple
from .rsbm_sampler import sample_rsbm_heun, sample_rsbm_euler
from .eval.evaluator import Evaluator
from .procrustes import fit_transformations
from .utils.logging import MetricsMeter, log_metrics_on_step, log_metrics_on_epoch


class RSBMPointFlow(RectifiedPointFlow):
    """
    ε-Rectified Schrödinger Bridge Matching for Point Cloud Assembly.

    Drop-in replacement for RectifiedPointFlow with RSBM's bridge kernel.
    Only the flow matching core is changed; everything else (encoder, DiT,
    data pipeline, evaluator) is inherited from RPF.
    """

    def __init__(
        self,
        feature_extractor: L.LightningModule,
        flow_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        encoder_ckpt: str = None,
        flow_model_ckpt: str = None,
        frozen_encoder: bool = False,
        anchor_free: bool = True,
        loss_type: str = "mse",
        timestep_sampling: str = "u_shaped",
        # ======== RSBM-specific parameters ========
        epsilon: float = 0.5,
        sigma_max: float = 1.0,
        inference_sampling_steps: int = 3,       # RSBM default: 3 steps!
        inference_sampler: str = "heun",          # RSBM default: Heun
        sigma_schedule: str = "linear",           # sigma schedule for sampling
        # ==========================================
        n_generations: int = 1,
        pred_proc_fn: Callable | None = None,
        save_results: bool = False,
    ):
        # Initialize parent RPF with all shared parameters
        super().__init__(
            feature_extractor=feature_extractor,
            flow_model=flow_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            encoder_ckpt=encoder_ckpt,
            flow_model_ckpt=flow_model_ckpt,
            frozen_encoder=frozen_encoder,
            anchor_free=anchor_free,
            loss_type=loss_type,
            timestep_sampling=timestep_sampling,
            inference_sampling_steps=inference_sampling_steps,
            inference_sampler=inference_sampler,
            n_generations=n_generations,
            pred_proc_fn=pred_proc_fn,
            save_results=save_results,
        )

        # ======== RSBM core ========
        self.epsilon = epsilon
        self.sigma_max = sigma_max
        self.sigma_schedule = sigma_schedule

        # Initialize RSBM bridge (replaces RPF's linear interpolation)
        self.rsbm_bridge = RSBMPointCloudBridgeSimple(epsilon=epsilon)

        self.save_hyperparameters(ignore=["feature_extractor", "flow_model"])

    # ------------------------------------------------------------------
    #  Override: timestep sampling (clamp upper bound for RSBM)
    # ------------------------------------------------------------------

    def _sample_timesteps(self, batch_size: int, **kwargs):
        """
        Sample timesteps with RSBM-safe upper bound.

        RPF uses t ∈ [eps, 1.0], but RSBM's bridge has a singularity at t=1
        (and t=0) where σ_t=0 and d(log σ)/dt diverges. We clamp the upper
        bound to 1-t_eps to avoid this.

        The bridge's compute_flow_target also clamps internally, but clamping
        here ensures the model never sees the exact boundary timestep.
        """
        t = super()._sample_timesteps(batch_size, **kwargs)
        # Clamp upper bound: RSBM bridge is degenerate at t=1 (σ=0)
        t = t.clamp(max=1.0 - self.rsbm_bridge.t_eps)
        return t

    # ------------------------------------------------------------------
    #  Override: flow target computation (THE critical replacement)
    # ------------------------------------------------------------------

    def _compute_flow_target(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Compute RSBM bridge sample and velocity target.

        REPLACES RPF's linear interpolation:
            RPF:  x_t = (1-t)·x_0 + t·x_1,  v_t = x_1 - x_0  (constant)
            RSBM: x_t = μ_t + σ_{ε,t}·noise, v_t = ∂μ/∂t + ∂(log σ)/∂t · residual  (time-varying)

        Args:
            x_0: (B, N, 3) ground truth assembled point cloud
            x_1: (B, N, 3) noise source
            t:   (B,) timesteps ∈ [eps, 1.0]

        Returns:
            x_t: (B, N, 3) bridge-interpolated point cloud
            v_t: (B, N, 3) conditional velocity target
        """
        return self.rsbm_bridge.compute_flow_target(x_0, x_1, t)

    # ------------------------------------------------------------------
    #  Override: loss computation (add RSBM-specific loss weighting)
    # ------------------------------------------------------------------

    def loss(self, output_dict: dict):
        """
        Compute RSBM velocity matching loss.

        Uses the same MSE/L1/Huber as RPF, but the target v_t is now
        RSBM's time-varying conditional velocity instead of constant (x_1 - x_0).

        Optionally applies time-dependent loss weighting to emphasize
        small-t (detail refinement) or large-t (global structure) steps.
        """
        v_pred = output_dict["v_pred"]
        v_t = output_dict["v_t"]

        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "huber":
            loss = F.huber_loss(v_pred, v_t, reduction="mean")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return {
            "loss": loss,
            "norm_v_pred": v_pred.norm(dim=-1).mean(),
            "norm_v_t": v_t.norm(dim=-1).mean(),
        }

    # ------------------------------------------------------------------
    #  Override: sampling (use Heun instead of Euler)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_rectified_flow(
        self,
        data_dict: dict,
        latent: dict,
        x_1: torch.Tensor | None = None,
        return_tarjectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Sample from RSBM using Heun ODE solver.

        Key difference from RPF:
          RPF:  50 Euler steps  (NFE = 50)
          RSBM: 3 Heun steps   (NFE = 5)

        The Heun (2nd-order) method gives:
            d1 = v(x_t, t)
            x̃  = x_t + d1·Δt
            d2 = v(x̃,  t+Δt)
            x_{t+1} = x_t + 0.5·(d1+d2)·Δt

        Args:
            data_dict: Input data dictionary
            latent: Feature latent dictionary from encoder
            x_1: Optional initial noise. If None, generates Gaussian noise.
            return_tarjectory: Whether to return all intermediate states.

        Returns:
            Final predicted point cloud (B, N, 3) or trajectory list.
        """
        anchor_indices = data_dict["anchor_indices"]
        scales = data_dict["scales"]

        def _flow_model_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            """Wrapper to call PointCloudDiT with scalar timestep."""
            B = x.shape[0]
            timesteps = torch.full((B,), t, device=x.device)
            return self.flow_model(
                x=x,
                timesteps=timesteps,
                latent=latent,
                scales=scales,
                anchor_indices=anchor_indices,
            )

        x_0 = data_dict["pointclouds_gt"]
        x_1 = torch.randn_like(x_0) if x_1 is None else x_1

        # Select sampler
        if self.inference_sampler == "heun":
            sampler_fn = sample_rsbm_heun
        elif self.inference_sampler == "euler":
            sampler_fn = sample_rsbm_euler
        else:
            # Fallback to RPF's built-in samplers for comparison
            from .sampler import get_sampler
            return get_sampler(self.inference_sampler)(
                flow_model_fn=_flow_model_fn,
                x_1=x_1,
                x_0=x_0,
                anchor_indices=anchor_indices if not self.anchor_free else None,
                num_steps=self.inference_sampling_steps,
                return_trajectory=return_tarjectory,
            )

        result = sampler_fn(
            flow_model_fn=_flow_model_fn,
            x_1=x_1,
            x_0=x_0,
            anchor_indices=anchor_indices if not self.anchor_free else None,
            num_steps=self.inference_sampling_steps,
            sigma_schedule=self.sigma_schedule,
            sigma_min=self.rsbm_bridge.t_eps,   # MUST match training range!
            sigma_max=self.sigma_max,
            return_trajectory=return_tarjectory,
        )
        return result
