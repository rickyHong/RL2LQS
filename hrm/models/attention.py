"""
Attention mechanisms for HRM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional relative positioning"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 use_relative_position: bool = False,
                 max_relative_position: int = 32):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_position = use_relative_position
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_relative_position:
            self.max_relative_position = max_relative_position
            self.relative_position_embeddings = nn.Embedding(
                2 * max_relative_position + 1, self.d_k
            )
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)  
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Attention mask of shape (batch_size, seq_len_q, seq_len_k)
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, seq_len_q, seq_len_k
        )
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.w_o(attention_output)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def scaled_dot_product_attention(self,
                                   Q: torch.Tensor,
                                   K: torch.Tensor, 
                                   V: torch.Tensor,
                                   mask: Optional[torch.Tensor],
                                   seq_len_q: int,
                                   seq_len_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position embeddings if enabled
        if self.use_relative_position:
            relative_scores = self.get_relative_position_scores(seq_len_q, seq_len_k)
            scores = scores + relative_scores
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_relative_position_scores(self, seq_len_q: int, seq_len_k: int) -> torch.Tensor:
        """Compute relative position scores"""
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift values to be >= 0
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get embeddings
        embeddings = self.relative_position_embeddings(final_mat)
        
        # Compute scores
        scores = torch.einsum('bhqd,qkd->bhqk', 
                            torch.zeros(1, self.num_heads, seq_len_q, self.d_k), 
                            embeddings)
        
        return scores


class CrossAttention(nn.Module):
    """Cross-attention for hierarchical communication"""
    
    def __init__(self,
                 query_dim: int,
                 key_value_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_k = query_dim // num_heads
        
        self.w_q = nn.Linear(query_dim, query_dim)
        self.w_k = nn.Linear(key_value_dim, query_dim)
        self.w_v = nn.Linear(key_value_dim, query_dim)
        self.w_o = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query: torch.Tensor,
                key_value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query tensor from one hierarchy level
            key_value: Key-value tensor from another hierarchy level
            mask: Optional attention mask
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key_value.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key_value).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(key_value).view(batch_size, seq_len_kv, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, -1
        )
        
        return self.w_o(output)


class SelfAttention(nn.Module):
    """Self-attention with optional causal masking"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.causal = causal
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        if self.causal:
            seq_len = x.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            if mask is not None:
                mask = mask * causal_mask
            else:
                mask = causal_mask
        
        output, _ = self.attention(x, x, x, mask)
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HierarchicalAttention(nn.Module):
    """Specialized attention for hierarchical reasoning"""
    
    def __init__(self,
                 high_level_dim: int,
                 low_level_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # High-level to low-level attention
        self.h2l_attention = CrossAttention(
            low_level_dim, high_level_dim, num_heads, dropout
        )
        
        # Low-level to high-level attention
        self.l2h_attention = CrossAttention(
            high_level_dim, low_level_dim, num_heads, dropout
        )
        
        # Self-attention for each level
        self.high_self_attention = SelfAttention(high_level_dim, num_heads, dropout)
        self.low_self_attention = SelfAttention(low_level_dim, num_heads, dropout)
    
    def forward(self,
                high_level_state: torch.Tensor,
                low_level_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            high_level_state: High-level state tensor
            low_level_state: Low-level state tensor
        
        Returns:
            Updated high-level and low-level states
        """
        # Self-attention within each level
        high_updated = self.high_self_attention(high_level_state)
        low_updated = self.low_self_attention(low_level_state)
        
        # Cross-attention between levels
        high_from_low = self.l2h_attention(high_updated, low_updated)
        low_from_high = self.h2l_attention(low_updated, high_updated)
        
        # Combine with residual connections
        high_final = high_updated + high_from_low
        low_final = low_updated + low_from_high
        
        return high_final, low_final