import torch
import torch.nn as nn
from einops import einsum
import math
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight: torch.Tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.W = nn.Parameter(data=weight)
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = einsum(x, self.W.T, '... d_in, d_in d_out -> ... d_out')
        return Y


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight: torch.Tensor = torch.empty(vocab_size, embedding_dim, device=device, dtype=dtype)
        self.embedding_dim = embedding_dim
        self.emb = nn.Parameter(data=weight)
    
    def forward(self, x: Int[Tensor, '...']) -> Float[Tensor, '... embedding_dim']:
        return self.emb[x]
        

class RMSNorm(torch.nn.Module):
    """"""
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.g
        return x_norm.to(in_dtype)


class FFN(torch.nn.Module):
    """
    FFN = (SiLU(x @ W1) * x @ W3) @ W2
    SiLU(x) = x * sigmoid(x)
    1. W1: (d_ff, d_model)
    2. W2: (d_model, d_ff)
    3. W3: (d_ff, d_model)
    4. d_ff: dimension of the feedforward layer
    5. d_model: dimension of the model
    d_ff = 8/3 * d_model
    """
    
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        hidden1 = einsum(x, self.W1.T, '... d_model, d_model d_ff -> ... d_ff')
        hidden2 = einsum(x, self.W3.T, '... d_model, d_model d_ff -> ... d_ff')
        silu = torch.sigmoid(hidden1) * hidden1
        hidden = silu * hidden2
        return einsum(hidden, self.W2.T, '... d_ff, d_ff d_model -> ... d_model')


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # build the rotary positional embeddings
        cos, sin = self._get_cos_sin(max_seq_len, theta, d_k, device=device) # (max_seq_len, d_k // 2)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def _get_cos_sin(self, max_seq_len: int, theta: float, d_k: int, device=None):
        k = torch.arange(0, d_k // 2, device=device)
        # rotation wave lengths
        # for theta = 10000, d_k = 512, we have wave length from 1~100.
        wave_length: Float[Tensor, 'd_k // 2'] = theta ** (-2 * k / d_k) # pyright: ignore[reportInvalidTypeForm]
        # thetas: Float[Tensor, 'max_seq_len d_k // 2'] = torch.outer(torch.arange(0, max_seq_len, device=device), wave_length) # type: ignore
        thetas = einsum(torch.arange(0, max_seq_len, device=device), wave_length, 's, d_k_half -> s d_k_half')
        return thetas.cos(), thetas.sin()


    def forward(self, x: Float[Tensor, "... seq_length d_k"], token_positions: Int[Tensor, "... seq_length"]=None) -> Float[Tensor, "... seq_length d_k"]:
        """Apply RoPE to the input tensor x.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Tensor of shape (..., seq_len) indicating the position of each token in the sequence.
        
        Returns:
            Tensor of the same shape as x with RoPE applied.
        """
        seq_len = x.size(-2)
        if token_positions is None:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
        else:
            cos = self.cos[token_positions]
            sin = self.sin[token_positions]
        x_pairs = rearrange(x, '... seq_len (d_k_half t) -> ... seq_len d_k_half t', t=2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        row1 = x1 * cos - x2 * sin
        row2 = x1 * sin + x2 * cos
        x_rotated = torch.stack([row1, row2], dim=-1)
        return rearrange(x_rotated, '... seq_len d_k_half t -> ... seq_len (d_k_half t)')


def softmax(x: Float[Tensor, '...'], dim: int) -> Float[Tensor, '...']:
    """A stable softmax implementation that prevents overflow/underflow.

    Args:
        x: Input tensor of shape (...,).
        i: The dimension along which to apply softmax.

    Returns:
        Tensor of the same shape as x with softmax applied along dimension i.
    """
    x_max = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max.values)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: Float[Tensor, "batch_size ... seq_len d_k"],
                      key: Float[Tensor, "batch_size ... seq_len d_k"],
                      value: Float[Tensor, "batch_size ... seq_len d_v"],
                      mask: Int[Tensor, "batch_size ... seq_len seq_len"]=None) -> Float[Tensor, "batch_size ... seq_len d_v"]:
        """Compute the scaled dot-product attention.

        Args:
            query: Query tensor of shape (..., seq_len, d_k)
            key: Key tensor of shape (..., seq_len, d_k)
            value: Value tensor of shape (..., seq_len, d_v)
            mask: Optional mask tensor of shape (..., seq_len, seq_len) containing 0s and 1s.

        Returns:
            Tensor of shape (..., seq_len_q, d_v) containing the attention output.
        """
        d_k = query.size(-1)
        
        qk = einsum(query, key, '... s1 d_k, ... s2 d_k -> ... s1 s2')
        scaled_qk = qk / math.sqrt(d_k)
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == False, float('-inf'))

        attn = softmax(scaled_qk, dim=-1)
        return einsum(attn, value, '... s1 s2, ... s2 d_v -> ... s1 d_v')
    

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # self.W = nn.Parameter(torch.empty(3 * d_model, d_model, device=device, dtype=dtype))
        # self.Wo = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.qkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)
        self.sdpa = ScaledDotProductAttention()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        """Compute the multi-head self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) containing the attention output.
        """
        seq_len = x.size(1)
        # Q, K, V = einsum(x, self.W.T, 'b s d1, d1 d2 -> b s d2').chunk(3, dim=-1)
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        
        Q = rearrange(Q, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)
        K = rearrange(K, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)
        V = rearrange(V, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        attn = self.sdpa(Q, K, V, mask)
        attn = rearrange(attn, '(b n_heads) s d_h -> b s (n_heads d_h)', n_heads=self.num_heads)
        return self.out(attn)
        # return einsum(attn, self.Wo.T, 'b s d_model, d_model d_model -> b s d_model')


class CausalMultiheadAttentionoWithRope(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # self.W = nn.Parameter(torch.empty(3 * d_model, d_model, device=device, dtype=dtype))
        # self.Wo = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        self.qkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.out = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        self.sdpa = ScaledDotProductAttention()

    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"], 
                token_positions: Int[Tensor, "batch_size seq_len"]=None) -> Float[Tensor, "batch_size seq_len d_model"]:
        """Compute the multi-head self-attention with RoPE.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Tensor of shape (batch_size, seq_len) indicating the position of each token in the sequence.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) containing the attention output.
        """
        seq_len = x.size(1)
        # Q, K, V = einsum(x, self.W.T, 'b s d1, d1 d2 -> b s d2').chunk(3, dim=-1)
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        
        Q = rearrange(Q, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)
        K = rearrange(K, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)
        V = rearrange(V, 'b s (n_heads d_h) ->  (b n_heads) s d_h', n_heads=self.num_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        attn = self.sdpa.forward(Q, K, V, mask)
        attn = rearrange(attn, '(b n_heads) s d_h -> b s (n_heads d_h)', n_heads=self.num_heads)
        return self.out(attn)
        # return einsum(attn, self.Wo.T, 'b s d_model, d_model d_model -> b s d_model')


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.mha = CausalMultiheadAttentionoWithRope(d_model=d_model, num_heads=num_heads,
                                                     theta=theta, max_seq_len=max_seq_len,
                                                     device=device, dtype=dtype)
        self.rms1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        """Compute the transformer block.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Tensor of shape (batch_size, seq_len) indicating the position of each token in the sequence.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) containing the output of the transformer block.
        """
        x = x + self.mha(self.rms1(x))
        x = x + self.ffn(self.rms2(x))
        return x
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
                 rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_rms = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, 'batch_size seq_len']) -> Float[Tensor, 'batch_size seq_len vocab_size']:
        """Compute the transformer output logits.
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, vocab_size) containing the output logits.
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_rms(x)
        logits = self.lm_head(x)
        return logits

