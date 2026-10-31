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

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # x (32, 1, 28, 28)
    h = F.conv2d(x, params["conv_in_w"], params["conv_in_b"], padding=1)
    # (32, 16, 28, 28)
    time_dim = params["time_mlp_w"].shape[1]
    temb = timestep_embedding(t, time_dim)
    # (B, 32), 32 dimensions for position

    temb = F.relu(
        F.linear(temb, params["time_mlp_w"], params["time_mlp_b"])
        )
    # time_mlp_w (16, 32)
    # time_mlp_b (16,)
    #temb (B:, 16)

    h += temb[:, :, None, None] 
    h = F.relu(h)

    h = F.relu(
        F.conv2d(
            h, params["conv_mid_w"], params["conv_mid_b"], padding=1
        )
    )
    
    return F.conv2d(h, params["conv_out_w"], params["conv_out_b"], padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):

    # (n, 1, size, size) (batch, channels, height, width)
    
    torch.manual_seed(seed)
    radius = size // 4 

    images = torch.zeros(n, 1, size, size)

    for i in range(n):
        # random center
        center = torch.randint(radius, size - radius, (2,))
        cx, cy = center

        for x in range(size):
            for y in range(size):
                # circle
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    images[i, 0, x, y] = 1.0

    return images

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    B = x0.shape[0]
    T = schedule["T"]
    
    torch.manual_seed(seed)

    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(x0)

    model = lambda x, t: tiny_unet_forward(x, t, params)
    loss = diffusion_training_loss(
        model,
        x0,
        t, 
        noise,
        schedule["alphas_cumprod"]
    )

    new_params = {}

    loss.backward()
    for name, p in params.items():
        if p.grad is not None:
            p_new = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            p_new = p.clone()
        new_params[name] = p_new

    return new_params, float(loss)

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:

    history = []

    for step in range(num_steps):
        torch.manual_seed(seed + step)

        indices = torch.randint(
            0, len(dataset), (batch_size,)
        )

        batch = dataset[indices]

        params, loss = ddpm_train_step(
            params, batch, schedule, 
            lr=lr,
            seed=seed + step)
        
        history.append(loss)
    
    return params, history

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

