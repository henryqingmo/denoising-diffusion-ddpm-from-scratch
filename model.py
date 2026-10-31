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

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # invert the q_sample equation for x0

    # select corresopnding alpha bar for each timestamp of image
    alpha_bar = extract_into_batch(alphas_cumprod, t, x_t)

    x0_hat = (
        x_t - torch.sqrt(1 - alpha_bar) * eps
        ) / torch.sqrt(alpha_bar)
    
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # return (posterior_mean, variance, x0_hat)
    alphas = schedule["alphas"]
    betas = schedule["betas"]
    alphas_cumprod = schedule["alphas_cumprod"]

    alpha_t = extract_into_batch(alphas, t, x_t)
    beta_t = extract_into_batch(betas, t, x_t)
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)
    
    # [1, .... alpha_cumprod[t - 1]]
    alphas_cumprod_prev = torch.cat([
        torch.ones_like(alphas_cumprod[:1]),
        alphas_cumprod[:-1]
    ])

    alpha_bar_prev = extract_into_batch(
        alphas_cumprod_prev, t, x_t
    )

    x0_hat = predict_x0_from_eps(
        x_t, t, eps, alphas_cumprod
        ).clamp(-1, 1)

    coef1 = (
        torch.sqrt(alpha_bar_prev) * beta_t 
        / (1 - alpha_bar_t)
    )

    coef2 = (
        torch.sqrt(alpha_t) * (1 - alpha_bar_prev)
        / (1 - alpha_bar_t)
    )

    mean = coef1 * x0_hat + coef2 * x_t
    variance = beta_t

    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # one reverse step x_t -> x_{t-1}
    
    if noise is None:
        noise = torch.randn_like(x_t)
    
    eps = tiny_unet_forward(x_t, t, params)

    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)

    # We want t = 0 to be deterministic, noise = 0
    mask = (t != 0).reshape(-1, 1, 1, 1)
    noise = noise * mask

    x_prev = mean + torch.sqrt(var) * noise

    return x_prev

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # ancestral sampling from pure noise to x0
    torch.manual_seed(seed)
    x = torch.randn(shape)
    T = schedule["T"]
    B = shape[0]

    for t in range(T-1, -1, -1):
        # All images shares the same timestamp 
        t_batch = torch.full((B,), t, dtype=torch.long)
        x = ddpm_p_sample(x, t_batch, params, schedule)

    return x

# Step 19 - sample_quality_mse
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    # mean over samples of min MSE to any dataset image
    res = []

    # for each sample, find the corresponding image with the minimum MSE
    for sample in samples:
        errors = [
            F.mse_loss(sample, image)
            for image in dataset
        ]

        min_error = torch.min(torch.stack(errors))
        res.append(min_error)
    
    return torch.stack(res).mean().item()

# Step 20 - ddpm_experiment
import torch
import torch.nn.functional as F

def ddpm_experiment(n_data: int = 64, size: int = 8, T: int = 20, hidden: int = 16, num_steps: int = 40, batch_size: int = 16, lr: float = 5e-2, n_samples: int = 8, seed: int = 0) -> dict:
    # data -> train -> sample -> metrics

    # 1. Create training images
    dataset = make_blob_dataset(n_data, size, seed)

    # 2. Create diffusion schedule
    schedule = build_diffusion_schedule(T)

    # 3. Initialize neural network
    params = init_tiny_unet(
        1, hidden, time_dim=hidden, seed=seed
    )

    # 4. Train DDPM
    params, history = train_ddpm(
        dataset, params, schedule,
        num_steps=num_steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed
    )

    # 5. Generate images using trained parameters
    shape = (n_samples, 1, size, size)

    samples = ddpm_sample_loop(
        params, schedule, shape, seed=seed + 1
    )

    # 6. Generate random noise as a baseline
    torch.manual_seed(seed + 2)
    noise_samples = torch.randn(shape)

    # 7. Evaluate both sets of images
    sample_mse = sample_quality_mse(samples, dataset)
    noise_mse = sample_quality_mse(noise_samples, dataset)

    improvement = noise_mse - sample_mse

    return {
        "train_losses": history,
        "final_loss": float(history[-1]),
        "sample_mse": sample_mse,
        "noise_mse": noise_mse,
        "improvement": improvement
    }

