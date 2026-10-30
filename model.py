"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    return a[t].reshape(-1, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    t #represent each image's timestamp 
    alpha_bar = extract_into_batch(alphas_cumprod, t, x0)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return F.mse_loss(noise_pred, noise)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    B = t.shape[0]
    half = dim // 2
    res = torch.zeros(B, dim, device=t.device)

    for b in range(B):
        for i in range(half):
            angle = t[b] / (10000 ** (2 * (i / half)))

            half = dim // 2
            res[b, i] = torch.sin(angle)
            res[b, i + half] = torch.cos(angle)

    return res

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    # Input convolution
    conv_in_w = (torch.randn(hidden, in_ch, 3, 3) * 0.02).requires_grad_()
    conv_in_b = torch.zeros(hidden, requires_grad=True)

    # Timestep embedding projection
    time_mlp_w = (torch.randn(hidden, time_dim) * 0.02).requires_grad_()
    time_mlp_b = torch.zeros(hidden, requires_grad=True)

    # Middle convolution
    conv_mid_w = (torch.randn(hidden, hidden, 3, 3) * 0.02).requires_grad_()
    conv_mid_b = torch.zeros(hidden, requires_grad=True)

    # Output convolution
    conv_out_w = (torch.randn(in_ch, hidden, 3, 3) * 0.02).requires_grad_()
    conv_out_b = torch.zeros(in_ch, requires_grad=True)

    return {
        "conv_in_w": conv_in_w,
        "conv_in_b": conv_in_b,
        "time_mlp_w": time_mlp_w,
        "time_mlp_b": time_mlp_b,
        "conv_mid_w": conv_mid_w,
        "conv_mid_b": conv_mid_b,
        "conv_out_w": conv_out_w,
        "conv_out_b": conv_out_b,
    }

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

