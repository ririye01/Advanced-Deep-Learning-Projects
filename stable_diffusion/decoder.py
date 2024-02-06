# External
import torch
from torch.nn import functional as F

# Internal
from attention import SelfAttention

class VAE_AttentionBlock(torch.nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.groupnorm = torch.nn.GroupNorm(32, channels)
    self.attn = SelfAttention(1, channels)

  # Forward Pass
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Store Residual
    residue = x

    # Apply Group Norm
    x = self.groupnorm(x)

    # x: (Batch Size, Channels, Height, Width)
    n, c, h, w = x.shape
    x = x.view((n, c, h * w)).transpose(-1, -2)

    # Attention
    x = self.attn(x)
    x = x.transpose(-1, -2).view((n, c, h, w))

    # Add Residual
    x += residue

    # Return
    return x
  
class VAE_ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super().__init__()
    # In Channels
    self.groupnorm_1 = torch.nn.GroupNorm(32, in_channels)
    self.conv_1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

    # Out Channels
    self.groupnorm_2 = torch.nn.GroupNorm(32, out_channels)
    self.conv_2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

    # Skip Connection
    if in_channels == out_channels:
      self.skip = torch.nn.Identity()
    else:
      self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)

  # Forward Pass
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Residual
    residue = x 

    # Norm + Activation + In Conv
    x = self.groupnorm_1(x)
    x = F.silu(x)
    x = self.conv_1(x)

    # Norm + Activation + Out Conv
    x = self.groupnorm_2(x)
    x = F.silu(x)
    x = self.conv_2(x)

    # Skip Connection
    return x + self.skip(residue)

class VAE_Decoder(torch.nn.Sequential):
  def __init__(self):
    super().__init__(
      torch.nn.Conv2d(4, 4, 1, 1, 0),
      torch.nn.Conv2d(4, 512, 3, 1, 1),

      # Attention Block
      VAE_ResidualBlock(512, 512),
      VAE_AttentionBlock(512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),

      # Upsample
      torch.nn.Upsample(scale_factor = 2),
      torch.nn.Conv2d(512, 512, 3, 1, 1),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      VAE_ResidualBlock(512, 512),
      torch.nn.Upsample(scale_factor = 2),
      torch.nn.Conv2d(512, 512, 3, 1, 1),
      VAE_ResidualBlock(512, 256),
      VAE_ResidualBlock(256, 256),
      VAE_ResidualBlock(256, 256),
      torch.nn.Upsample(scale_factor = 2),
      torch.nn.Conv2d(256, 256, 3, 1, 1),
      VAE_ResidualBlock(256, 128),
      VAE_ResidualBlock(128, 128),
      VAE_ResidualBlock(128, 128),

      # Norm + Activation
      torch.nn.GroupNorm(32, 128),
      torch.nn.SiLU(),
      
      # Output
      torch.nn.Conv2d(128, 3, 3, 1, 1),
    )

  # Forward Pass
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Reverse Scaling
    x /= 0.18215
    for module in self:
      x = module(x)
    return x