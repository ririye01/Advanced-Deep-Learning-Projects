import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt = None,
    input_image = None,
    strength = 0.8,
    do_cfg = True,
    cfg_scale = 7.5,
    sampler_name = "ddpm",
    n_inference_steps = 50,
    models = {},
    seed = None,
    device = None,
    idle_device = None,
    tokenizer = None,
):
    # Inferencing Model
    with torch.no_grad():
        if not (0 < strength <= 1.0):
            raise ValueError("Strength Must Be Between (0, 1]")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)