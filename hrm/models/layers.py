"""
Basic neural network layers for HRM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention and feed-forward"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask
            key_padding_mask: Key padding mask
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(
            x, x, x, 
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    
    def __init__(self, 
                 d_model: int,
                 d_hidden: int,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for controlling information flow"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_value = nn.Linear(d_model, d_ff)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.linear_gate(x))
        value = self.linear_value(x)
        return gate * value


class StateProjection(nn.Module):
    """Project between different state dimensions"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 use_bias: bool = True,
                 activation: Optional[str] = None):
        super().__init__()
        
        self.projection = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.projection(x))


class RecurrentCell(nn.Module):
    """Custom recurrent cell for hierarchical processing"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden transformation
        self.ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        # Hidden-to-hidden transformation  
        self.hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
    def forward(self, 
                input: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state of shape (batch_size, hidden_size)
        
        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        if hidden is None:
            hidden = torch.zeros(
                input.size(0), self.hidden_size,
                dtype=input.dtype, device=input.device
            )
        
        # Compute gates
        gi = self.ih(input)
        gh = self.hh(hidden)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        
        # Reset and update gates
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        
        # New gate
        new = torch.tanh(i_n + reset * h_n)
        
        # Update hidden state
        hidden = (1 - update) * new + update * hidden
        
        return hidden


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]