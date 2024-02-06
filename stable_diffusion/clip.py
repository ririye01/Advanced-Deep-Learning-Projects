# External
import torch
from torch.nn import functional as F

# Internal
from attention import SelfAttention

class CLIPEmbedding(torch.nn.Module):
  def __init__(self, n_vocab, d_embed, n_tokens):
    super().__init__()
    self.token_embedding = torch.nn.Embedding(n_vocab, d_embed)
    self.position_embedding = torch.nn.Parameter(torch.zeros(n_tokens, d_embed)) # Learnable Position Embedding

  # Add Token Embedding, Position Embedding
  def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
    x = self.token_embedding(tokens)
    x += self.position_embedding
    return x
  
class CLIPLayer(torch.nn.Module):
  def __init__(self, n_heads, d_embed):
    super().__init__()
    self.layernorm_1 = torch.nn.LayerNorm(d_embed)
    self.attn = SelfAttention(n_heads, d_embed)
    self.layernorm_2 = torch.nn.LayerNorm(d_embed)
    self.linear_1 = torch.nn.Linear(d_embed, d_embed * 4)
    self.linear_2 = torch.nn.Linear(d_embed * 4, d_embed)

  # Prenorm (See Transformer Architecture)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Self Attn
    residue = x
    x = self.layernorm_1(x)
    x = self.attn(x, causal_mask = True)
    x += residue

    # FFN
    residue = x
    x = self.layernorm_2(x)
    x = self.linear_1(x)
    x = x * torch.sigmoid(1.702 * x) # QuickGELU
    x = self.linear_2(x)
    x += residue
    return x

class CLIP(torch.nn.Module):
  def __init__(self):
    self.embedding = CLIPEmbedding(49408, 768, 77)
    self.layers = torch.nn.Module([
      CLIPLayer(12, 768) for i in range(12)
    ])
    self.layernorm = torch.nn.LayerNorm(768)

  def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
    tokens = tokens.type(torch.long)
    state = self.embedding(tokens)
    for layer in self.layers:
      state = layer(state)
    output = self.layernorm(state)
    return output
  