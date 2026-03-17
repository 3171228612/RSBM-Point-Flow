"""
ε-Rectified Schrödinger Bridge for 3D Point Cloud Registration.

This module adapts the RSBM bridge kernel from visual navigation (2D waypoints)
to 3D point cloud assembly/registration. Key adaptation:
  - Input: 3D point clouds (B, N, 3) instead of 2D trajectories (B, H, 2)
  - Bridge: connects condition point cloud x_0 (GT assembled) to noise x_1 (Gaussian)
  - ε parameter controls path straightness for few-step ODE sampling

The bridge interpolation follows the VE (Variance Exploding) formulation:
  μ_t  = s_t · x_1 + (1 - s_t) · x_0,   where s_t = t² / σ_max²
  σ_t  = √ε · t · √(1 - s_t)

When ε=1: standard Brownian Bridge (same as original SB)
When ε→0: deterministic OT interpolant (same as RPF's linear interpolation)
When ε∈(0,1): ε-rectified bridge — our sweet spot for few-step sampling.

Reference: "Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation"
"""

import torch


class RSBMPointCloudBridge:
    """
    ε-Rectified Schrödinger Bridge for 3D point cloud data.

    Provides:
      1. bridge_sample:           sample x_t ~ q_ε(x_t | x_0, x_1)
      2. conditional_velocity:    compute target v_t for velocity matching
      3. compute_flow_target:     drop-in replacement for RPF's _compute_flow_target
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        sigma_max: float = 1.0,
    ):
        """
        Args:
            epsilon: Bridge stochasticity parameter ∈ (0, 1].
                     1.0 = standard Brownian Bridge
                     <1.0 = straighter (rectified) paths
                     Recommended: 0.5 for point cloud assembly.
            sigma_max: Maximum noise level (normalizing constant for s_t).
                       For [0,1] time range, use 1.0.
        """
        self.epsilon = epsilon
        self.sigma_max = sigma_max

    def _get_schedule(self, t: torch.Tensor):
        """
        Compute bridge schedule quantities.

        Args:
            t: (B,) or (B, 1, 1) timesteps in [0, σ_max]

        Returns:
            s_t:         normalized time s = t² / σ_max²
            one_minus_s: 1 - s_t
            std_t:       bridge standard deviation √(ε · t² · (1 - s_t))
        """
        s_t = t ** 2 / self.sigma_max ** 2
        one_minus_s = 1.0 - s_t
        var_t = self.epsilon * (t ** 2) * one_minus_s
        std_t = torch.sqrt(torch.clamp(var_t, min=1e-12))
        return s_t, one_minus_s, std_t

    def bridge_mean(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        """
        Compute the bridge mean μ_t = s_t · x_1 + (1 - s_t) · x_0.

        Args:
            x_0: (B, N, 3) ground truth assembled point cloud
            x_1: (B, N, 3) noise / source point cloud
            t:   (B, 1, 1) timesteps

        Returns:
            mu_t: (B, N, 3) bridge mean
        """
        s_t = t ** 2 / self.sigma_max ** 2
        return s_t * x_1 + (1 - s_t) * x_0

    def bridge_sample(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Sample x_t from the ε-rectified bridge kernel q_ε(x_t | x_0, x_1).

        x_t = μ_t + σ_{ε,t} · ε,  where ε ~ N(0, I)

        Args:
            x_0: (B, N, 3) ground truth assembled point cloud
            x_1: (B, N, 3) noise source
            t:   (B,) timesteps
            noise: optional pre-sampled Gaussian noise

        Returns:
            x_t:   (B, N, 3) interpolated noisy point cloud
            noise: (B, N, 3) the noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        t_ = t.view(-1, 1, 1)  # (B, 1, 1)
        s_t, one_minus_s, std_t = self._get_schedule(t_)

        mu_t = s_t * x_1 + one_minus_s * x_0
        x_t = mu_t + std_t * noise
        return x_t, noise

    def conditional_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Compute the conditional velocity field of the ε-rectified bridge:

        v_t = dμ_t/dt + (d log σ_t / dt) · (x_t - μ_t)

        where:
            dμ_t/dt         = (2t / σ_max²) · (x_1 - x_0)
            d log σ_t / dt  = (1 - 2s_t) / (t · (1 - s_t))

        CRITICAL PROPERTY: d log σ_t / dt is ε-INVARIANT!
        The √ε factors cancel in the logarithmic derivative.
        This means the network learns the same velocity function for all ε,
        but ε controls the training sample distribution (path concentration).

        NUMERICAL NOTE: The score term d(log σ)/dt · (x_t - μ_t) is computed
        analytically as (1-2s)·√ε / √(1-s) · noise to avoid 0×∞ at boundaries.

        Args:
            x_0: (B, N, 3) ground truth
            x_1: (B, N, 3) noise
            x_t: (B, N, 3) current interpolated state
            t:   (B,) timesteps (should be clamped away from 0 and σ_max)
            noise: (B, N, 3) the noise used to sample x_t (for analytic computation)

        Returns:
            v_target: (B, N, 3) conditional velocity target
        """
        t_ = t.view(-1, 1, 1)  # (B, 1, 1)
        s_t = t_ ** 2 / self.sigma_max ** 2
        one_minus_s = 1.0 - s_t

        # dμ/dt = (2t/σ²_max)(x_1 - x_0)
        dmu_dt = (2 * t_ / self.sigma_max ** 2) * (x_1 - x_0)

        if noise is not None:
            # Analytic computation (numerically stable):
            # d(log σ)/dt · std · noise
            #   = [(1-2s)/(t(1-s))] · [√ε · t · √(1-s)] · noise
            #   = (1-2s) · √ε / √(1-s) · noise
            # Note: σ_max cancels out because s = t²/σ²_max.
            sqrt_one_minus_s = torch.sqrt(one_minus_s)
            sqrt_eps = self.epsilon ** 0.5
            score_term = (1 - 2 * s_t) * sqrt_eps / (sqrt_one_minus_s + 1e-8) * noise
        else:
            # Fallback: separate computation (may have numerical issues at boundaries)
            mu_t = s_t * x_1 + one_minus_s * x_0
            residual = x_t - mu_t
            dlog_std_dt = (1 - 2 * s_t) / (t_ * one_minus_s + 1e-8)
            score_term = dlog_std_dt * residual

        velocity = dmu_dt + score_term
        return velocity

    def compute_flow_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Drop-in replacement for RPF's _compute_flow_target.

        Instead of:
            x_t = (1-t)*x_0 + t*x_1      (RPF: linear interpolation)
            v_t = x_1 - x_0               (RPF: constant velocity)

        We compute:
            x_t = μ_t + σ_{ε,t}·noise     (RSBM: ε-rectified bridge sample)
            v_t = conditional velocity     (RSBM: time-varying velocity)

        Args:
            x_0: (B, N, 3) ground truth assembled point cloud
            x_1: (B, N, 3) noise source
            t:   (B,) timesteps ∈ [0, σ_max]

        Returns:
            x_t: (B, N, 3) bridge-interpolated point cloud
            v_t: (B, N, 3) conditional velocity target
        """
        # Clamp t to avoid boundary singularities at t=0 and t=σ_max
        t_eps = 0.005 * self.sigma_max
        t = t.clamp(t_eps, self.sigma_max - t_eps)

        x_t, noise = self.bridge_sample(x_0, x_1, t)
        v_t = self.conditional_velocity(x_0, x_1, x_t, t, noise=noise)
        return x_t, v_t


class RSBMPointCloudBridgeSimple:
    """
    Simplified RSBM bridge using [0, 1] time range (matches RPF's convention).

    In RPF, t ∈ [0, 1] where:
      t=1: noise (x_1)
      t=0: clean data (x_0)

    We adapt the VE bridge by setting σ_max = 1.0:
      s_t = t²
      μ_t = t² · x_1 + (1 - t²) · x_0
      σ_t = √ε · t · √(1 - t²)

    The velocity target becomes:
      dμ/dt = 2t · (x_1 - x_0)
      d(log σ)/dt = (1 - 2t²) / (t(1 - t²))
      v_t = dμ/dt + d(log σ)/dt · (x_t - μ_t)

    NUMERICAL NOTE:
      At t=0 and t=1, the bridge is degenerate (σ=0) and d(log σ)/dt diverges.
      We avoid this by clamping t to [t_eps, 1-t_eps] during training, and by
      computing the score term analytically:
        d(log σ)/dt · σ · noise = (1-2s) · √ε / √(1-s) · noise
      This avoids the 0×∞ indeterminate form from separate computation.
    """

    def __init__(self, epsilon: float = 0.5, t_eps: float = 0.05):
        """
        Args:
            epsilon: Bridge stochasticity ∈ (0, 1]. Lower = straighter paths.
            t_eps: Margin to avoid boundary singularities at t=0 and t=1.
                   The score term |(1-2t²)·√ε/√(1-t²)| diverges as t→1.
                   At t_eps=0.05: max coefficient ≈ 1.5 (safe).
                   At t_eps=0.005: max coefficient ≈ 6.9 (causes training instability).
        """
        self.epsilon = epsilon
        self.t_eps = t_eps
        self._sqrt_eps = epsilon ** 0.5

    def compute_flow_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Compute RSBM bridge sample and velocity target.

        Args:
            x_0: (B, N, 3) ground truth (assembled)
            x_1: (B, N, 3) noise
            t:   (B,) timesteps in [eps, 1.0]

        Returns:
            x_t: (B, N, 3) bridge sample
            v_t: (B, N, 3) velocity target
        """
        # ---- Clamp t to avoid boundary singularities ----
        # At t=0 and t=1, σ_t=0 and d(log σ)/dt diverges.
        # The original RSBM uses sigma_max-1e-4 in its schedule for the same reason.
        t = t.clamp(self.t_eps, 1.0 - self.t_eps)

        noise = torch.randn_like(x_0)
        t_ = t.view(-1, 1, 1)  # (B, 1, 1)

        # Schedule
        s = t_ ** 2                           # s_t = t²  (σ_max=1)
        one_minus_s = 1.0 - s
        std = self._sqrt_eps * t_ * torch.sqrt(one_minus_s)   # √ε · t · √(1-t²)

        # Bridge sample
        mu = s * x_1 + one_minus_s * x_0
        x_t = mu + std * noise

        # Velocity target — computed analytically to avoid 0×∞ at boundaries
        #
        # v_t = dμ/dt + d(log σ)/dt · (x_t - μ_t)
        #
        # where (x_t - μ_t) = std · noise = √ε · t · √(1-s) · noise
        #
        # The score term d(log σ)/dt · std · noise simplifies analytically:
        #   = [(1-2s)/(t·(1-s))] · [√ε · t · √(1-s)] · noise
        #   = (1-2s) · √ε / √(1-s) · noise
        #
        # This is well-defined for all t ∈ (0, 1) (after clamping).

        dmu_dt = 2 * t_ * (x_1 - x_0)

        # Analytic score term: (1-2s)·√ε / √(1-s) · noise
        sqrt_one_minus_s = torch.sqrt(one_minus_s)  # already > 0 after clamping
        score_term = (1.0 - 2.0 * s) * self._sqrt_eps / (sqrt_one_minus_s + 1e-8) * noise

        v_t = dmu_dt + score_term

        # Safety: clamp any residual NaN/Inf (shouldn't happen after above fixes)
        if torch.isnan(v_t).any() or torch.isinf(v_t).any():
            v_t = torch.nan_to_num(v_t, nan=0.0, posinf=0.0, neginf=0.0)

        return x_t, v_t
