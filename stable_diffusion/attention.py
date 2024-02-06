# External
import torch
from torch.nn import functional as F
import math

# Self Attention
class SelfAttention(torch.nn.Module):
  def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
    super().__init__()
    # Combine All Projections
    self.in_proj = torch.nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
    self.out_proj = torch.nn.Linear(d_embed, d_embed, bias = out_proj_bias)
    self.d_embed = d_embed
    self.n_heads = n_heads
    self.head_dim = d_embed // n_heads

  # Forward Pass
  def forward(self, x: torch.Tensor, causal_mask = False) -> torch.Tensor:
    # Extract Shape
    input_shape = x.shape
    batch_size, seq_len, d_embed = input_shape

    # Q, K, V
    interim_shape = (batch_size, seq_len, self.n_heads, self.head_dim)
    q, k, v = self.in_proj(x).chunk(3, dim = -1)
    q = q.view(interim_shape).transpose(1, 2)
    k = k.view(interim_shape).transpose(1, 2)
    v = v.view(interim_shape).transpose(1, 2)

    # QK^T
    weight = q @ k.transpose(-1, -2)
    
    # Mask (Fill Triu With -Inf)
    if causal_mask:
      mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
      weight.masked_fill_(mask, -torch.inf) 

    # Softmax
    weight /= math.sqrt(self.head_dim)
    weight = F.softmax(weight, dim = -1)

    # V, Transpose
    output = weight @ v
    output = output.transpose(1, 2).reshape(input_shape)
    output = self.out_proj(output)
    return output

# Cross Attention
class CrossAttention(torch.nn.Module):
  def __init__(self, n_heads, d_embed, d_cross, in_proj_bias = True, out_proj_bias = True):
    super().__init__()
    # K, V Crossed, Q Projected
    self.q_proj = torch.nn.Linear(d_embed, d_embed, bias = in_proj_bias)
    self.k_proj = torch.nn.Linear(d_cross, d_embed, bias = in_proj_bias)
    self.v_proj = torch.nn.Linear(d_cross, d_embed, bias = in_proj_bias)
    self.out_proj = torch.nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    self.d_embed = d_embed
    self.n_heads = n_heads
    self.head_dim = d_embed // n_heads

  # Forward Pass (x: Latent, cond: Conditioning / Context)
  def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tesnor:
    # Extract Shape
    input_shape = x.shape
    batch_size, seq_len, d_embed = input_shape

    # Divide Each Q Embedding Into Multiple Heads (n_heads * head_dim = dim_Q)
    interim_shape = (batch_size, -1, self.n_heads, self.head_dim)
    q = self.q_proj(x).view(interim_shape).transpose(1, 2)
    k = self.k_proj(cond).view(interim_shape).transpose(1, 2)
    v = self.v_proj(cond).view(interim_shape).transpose(1, 2)

    # Attention
    weight = q @ k.transpose(-1, -2)
    weight /= math.sqrt(self.head_dim)
    weight = F.softmax(weight, dim = -1)
    output = weight @ v

    # Contiguous => Memory Locations Close
    output = output.transpose(1, 2).contiguous().view(input_shape)

    # Output Projection
    output = self.out_proj(output)

    # Return
    return output
    