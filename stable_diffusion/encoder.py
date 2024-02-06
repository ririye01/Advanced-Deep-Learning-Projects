# External
import torch
from torch.nn import functional as F

# Internal
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# Compress Image To Latent Space
class VAE_Encoder(torch.nn.Sequential):
  def __init__(self):
    super().__init__(
      # Compression (In, Out, Kernel, Stride, Padding)
      torch.nn.Conv2d(3, 128, 3, 1, 1), 
      VAE_ResidualBlock(128, 128),
      VAE_ResidualBlock(128, 128),
      torch.nn.Conv2d(128, 128, 3, 2, 0),
      VAE_ResidualBlock(128, 256), # 128 -> 256
      VAE_ResidualBlock(256, 256),
      torch.nn.Conv2d(256, 256, 3, 2, 0),
      VAE_ResidualBlock(256, 512), # 256 -> 512
      VAE_ResidualBlock(512, 512),
      torch.nn.Conv2d(512, 512, 3, 2, 0),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),

      # Attention Block
      VAE_AttentionBlock(512),
      VAE_ResidualBlock(512, 512),

      # Norm + Activation
      torch.nn.GroupNorm(32, 512),
      torch.nn.SiLU(),

      # Bottleneck
      torch.nn.Conv2d(512, 8, 3, 1, 1),
      torch.nn.Conv2d(8, 8, 1, 1, 0),
    )

    # Forward Pass
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
      for module in self:
        if getattr(module, 'stride', None) == (2, 2):
          # Padding: Left, Right, Top, Bottom
          x = F.pad(x, (0, 1, 0, 1))
        x = module(x)

      # VAE Learns Mean, Log Variance
      mean, log_var = x.chunk(x, 2, dim = 1)

      # Clamp (Variance Between 1e-14, 1e8)
      log_var = torch.clamp(log_var, -30, 20)

      # Exponentiate
      var = log_var.exp()
      std = var.sqrt()

      # N(0, 1) -> N(mean, std)
      x = mean + std * noise

      # Scale Output
      x *= 0.18215

      # Return
      return x
