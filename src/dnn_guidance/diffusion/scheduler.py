from __future__ import annotations

import torch


class NoiseScheduler:
    """Linear beta schedule for DDPM training and sampling."""

    def __init__(self, beta_start: float = 1e-4, beta_end: float = 0.02, timesteps: int = 1000) -> None:
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion to ``clean`` at timestep ``t``."""
        sqrt_ab = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None, None]
        return sqrt_ab * clean + sqrt_one_minus_ab * noise

    def step(self, model_noise: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Perform one reverse diffusion step."""
        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars[t])[:, None, None, None]
        sqrt_recip_alpha = torch.sqrt(1.0 / alpha_t)
        pred = (x_t - sqrt_one_minus_ab * model_noise) / torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * model_noise)
        if (t == 0).all():
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise
