"""
RSBM Sampling Utilities for Point Cloud Registration.

Provides Heun (2nd-order) ODE sampler adapted for 3D point cloud data,
replacing RPF's Euler/RK2/RK4 samplers.

Key differences from RPF's sampler:
  1. Uses Heun's method (2nd order) — better accuracy per NFE than Euler
  2. Supports Karras sigma schedule (non-uniform time steps)
  3. Supports the bridge-aware ODE drift (score from denoised x_0)
  4. FSAL optimization: NFE = 2k - 1 for k steps

Integration direction: t goes from σ_max → 0 (high noise to clean data)
"""

import torch
from typing import Callable
from functools import partial


def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0, device="cpu"):
    """
    Karras sigma schedule — concentrates steps near small σ where details matter.

    Produces n sigmas from σ_max down to σ_min, plus a trailing 0.

    Args:
        n: number of steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level
        rho: schedule curvature (7.0 is Karras default)

    Returns:
        Tensor of shape (n+1,) with sigmas[0]=σ_max, ..., sigmas[n]=0
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


def get_sigmas_uniform(n: int, sigma_min: float, sigma_max: float, device="cpu"):
    """
    Uniform sigma schedule — linear spacing from σ_max to σ_min.

    Args:
        n: number of steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        Tensor of shape (n+1,) with trailing 0
    """
    sigmas = torch.linspace(sigma_max, sigma_min, n)
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


def get_sigmas_linear_01(n: int, eps: float = 0.01, device="cpu"):
    """
    Linear schedule in [0, 1] — matches RPF's time convention.

    Goes from t≈1.0 (noise) down to t=eps (near clean), with trailing 0.
    Starts at 1.0-eps (not exactly 1.0) to avoid the degenerate bridge point
    where σ_t=0 and the velocity target diverges.

    Args:
        n: number of steps
        eps: small positive to avoid t=0 and t=1 singularities

    Returns:
        Tensor of shape (n+1,) from (1-eps) down to 0
    """
    sigmas = torch.linspace(1.0 - eps, eps, n)
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


# ============================================================================
#  Bridge-aware ODE drift (for VE bridge with score-based formulation)
# ============================================================================

def rsbm_to_d_ve(x, sigma, denoised, x_1, sigma_max, w=1.0):
    """
    Compute probability flow ODE drift for VE bridge.

    d = -0.5 · g²_t · (∇log p(x_t|x_0) - w · ∇log p(x_1|x_t))

    where:
        ∇log p(x_t|x_0) = (denoised - x) / σ²
        ∇log p(x_1|x_t) = (x_1 - x) / (σ²_max - σ²)
        g²_t = 2σ

    Args:
        x: current state (B, N, 3)
        sigma: current noise level (B,)
        denoised: predicted clean data (B, N, 3)
        x_1: terminal noise (B, N, 3)
        sigma_max: max noise level
        w: guidance weight

    Returns:
        d: ODE drift (B, N, 3)
    """
    sigma_ = sigma.view(-1, 1, 1)
    sigma_max_sq = sigma_max ** 2

    score_x0 = (denoised - x) / (sigma_ ** 2)
    score_x1 = (x_1 - x) / (sigma_max_sq - sigma_ ** 2 + 1e-8)
    g_sq = 2 * sigma_
    d = -0.5 * g_sq * (score_x0 - w * score_x1)
    return d


# ============================================================================
#  Samplers
# ============================================================================

@torch.no_grad()
def sample_rsbm_heun(
    flow_model_fn: Callable,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    anchor_indices: torch.Tensor = None,
    num_steps: int = 3,
    sigma_schedule: str = "linear",
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    rho: float = 7.0,
    return_trajectory: bool = False,
):
    """
    RSBM Heun (2nd-order) ODE sampler for point cloud registration.

    Uses velocity prediction: the model directly outputs v(x_t, t),
    and we integrate dx = v · dt using Heun's method.

    Heun's method at each step:
        d1 = v(x_t, t)
        x̃ = x_t + d1 · Δt
        d2 = v(x̃, t + Δt)
        x_{t+1} = x_t + 0.5·(d1 + d2)·Δt

    This gives NFE = 2k - 1 for k steps (last step uses Euler).

    Args:
        flow_model_fn: function(x, t) -> velocity prediction (B, N, 3)
        x_1: (B, N, 3) initial noise
        x_0: (B, N, 3) ground truth (used only for anchor reset)
        anchor_indices: (B, N) bool mask for anchor points (if any)
        num_steps: number of ODE steps (default 3 for RSBM)
        sigma_schedule: "karras", "uniform", or "linear"
        sigma_min: minimum sigma
        sigma_max: maximum sigma
        rho: Karras schedule rho
        return_trajectory: if True, return all intermediate states

    Returns:
        x_0_pred: (B, N, 3) final prediction
        or trajectory: list of (B, N, 3) at each step
    """
    device = x_1.device

    # Build sigma schedule (from high to low, ending at 0)
    # CRITICAL: All schedules must stay within [sigma_min, 1-sigma_min]
    # to match the training range [t_eps, 1-t_eps]. The model has never
    # seen t > 1-t_eps, so querying outside causes garbage predictions.
    sigma_upper = sigma_max - sigma_min   # = 1.0 - t_eps = 0.95
    if sigma_schedule == "karras":
        sigmas = get_sigmas_karras(num_steps, sigma_min, sigma_upper, rho, device)
    elif sigma_schedule == "uniform":
        sigmas = get_sigmas_uniform(num_steps, sigma_min, sigma_upper, device)
    elif sigma_schedule == "linear":
        sigmas = get_sigmas_linear_01(num_steps, eps=sigma_min, device=device)
    else:
        raise ValueError(f"Unknown sigma schedule: {sigma_schedule}")

    x = x_1.clone()

    # Reset anchor points
    if anchor_indices is not None:
        x[anchor_indices] = x_0[anchor_indices]

    trajectory = [x.detach().clone()] if return_trajectory else None
    nfe = 0

    for i in range(len(sigmas) - 1):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        dt = t_next - t_cur  # negative (going from high t to low t)

        # Velocity at current point
        v1 = flow_model_fn(x, t_cur.item())
        nfe += 1

        if t_next == 0:
            # Last step: use Euler (no 2nd evaluation needed)
            x = x + v1 * dt
        else:
            # Heun's method: 2nd order correction
            x_euler = x + v1 * dt
            if anchor_indices is not None:
                x_euler[anchor_indices] = x_0[anchor_indices]

            v2 = flow_model_fn(x_euler, t_next.item())
            nfe += 1

            v_avg = 0.5 * (v1 + v2)
            x = x + v_avg * dt

        # Reset anchor points
        if anchor_indices is not None:
            x[anchor_indices] = x_0[anchor_indices]

        if return_trajectory:
            trajectory.append(x.detach().clone())

    if return_trajectory:
        return trajectory
    return x.detach()


@torch.no_grad()
def sample_rsbm_euler(
    flow_model_fn: Callable,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    anchor_indices: torch.Tensor = None,
    num_steps: int = 5,
    sigma_schedule: str = "linear",
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    rho: float = 7.0,
    return_trajectory: bool = False,
):
    """
    Euler (1st-order) ODE sampler — baseline for comparison.

    NFE = num_steps (one evaluation per step).
    """
    device = x_1.device

    # All schedules must stay within [sigma_min, 1-sigma_min] (= training range)
    sigma_upper = sigma_max - sigma_min
    if sigma_schedule == "karras":
        sigmas = get_sigmas_karras(num_steps, sigma_min, sigma_upper, rho, device)
    elif sigma_schedule == "uniform":
        sigmas = get_sigmas_uniform(num_steps, sigma_min, sigma_upper, device)
    elif sigma_schedule == "linear":
        sigmas = get_sigmas_linear_01(num_steps, eps=sigma_min, device=device)
    else:
        raise ValueError(f"Unknown sigma schedule: {sigma_schedule}")

    x = x_1.clone()
    if anchor_indices is not None:
        x[anchor_indices] = x_0[anchor_indices]

    trajectory = [x.detach().clone()] if return_trajectory else None

    for i in range(len(sigmas) - 1):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]
        dt = t_next - t_cur

        v = flow_model_fn(x, t_cur.item())
        x = x + v * dt

        if anchor_indices is not None:
            x[anchor_indices] = x_0[anchor_indices]

        if return_trajectory:
            trajectory.append(x.detach().clone())

    if return_trajectory:
        return trajectory
    return x.detach()


# ============================================================================
#  Sampler Factory
# ============================================================================

def get_rsbm_sampler(sampler_name: str, **kwargs):
    """
    Get RSBM sampler by name.

    Args:
        sampler_name: "heun" or "euler"
        **kwargs: additional kwargs passed to the sampler

    Returns:
        Sampler function with signature (flow_model_fn, x_1, x_0, anchor_indices, num_steps, ...)
    """
    samplers = {
        "heun": sample_rsbm_heun,
        "euler": sample_rsbm_euler,
    }
    if sampler_name not in samplers:
        raise ValueError(f"Unknown RSBM sampler: {sampler_name}. Available: {list(samplers.keys())}")
    return partial(samplers[sampler_name], **kwargs)
