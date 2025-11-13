import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
# from einops import rearrange

'''
class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):

        batch_size, seq_len = x.shape
        k = self.key(x).transpose(0, 1)
        q = self.query(x)
        v = self.value(x)

        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # import matplotlib.pyplot as plt
        # plt.plot(x[0, :, 0].detach().cpu().numpy())
        # plt.show()

        out = torch.matmul(attn, v)

        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out



class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class Attention_Rel_Vec(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size/num_heads)))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, self.num_heads, seq_len, seq_len)

        attn = torch.matmul(q, k)
        # attn shape (seq_len, seq_len)
        attn = (attn + Srel) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
'''

class CrossAttnTRMBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(CrossAttnTRMBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def compute_attention(self, rep_pair: List[torch.Tensor], mask: Optional[torch.Tensor] = None):
        """Compute attention output for the pair of representations"""
        rep_mask_token, rep_visible = rep_pair[1], rep_pair[0]
        return self.attn(rep_mask_token, rep_visible, rep_visible, mask=mask)
        
    def compute_ffn(self, x):
        """Apply feed-forward network to input"""
        return self.ffn(x)

    def forward(self, rep_visible, rep_mask_token, mask: Optional[torch.Tensor] = None):
        x = [rep_visible, rep_mask_token]
        # Compute the attention output first
        attn_output = self.compute_attention(x, mask=mask)
        # Apply skip connection with pre-computed output
        x = self.skipconnect1(x, attn_output)
        # Compute FFN output
        ffn_output = self.compute_ffn(x)
        # Apply second skip connection
        x = self.skipconnect2(x, ffn_output)
        return x

'''
class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
'''


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention' (TorchScript compatible)
    """
    def __init__(self, dropout=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, apply_dropout: bool = False):
        # Calculate attention scores with explicit scaling
        d_k = query.size(-1)
        scale = 1.0 / math.sqrt(float(d_k))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Handle mask with explicit condition checking to make it TorchScript friendly
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        # Apply dropout with explicit condition
        if apply_dropout:
            p_attn = self.dropout(p_attn)
        
        output = torch.matmul(p_attn, value)
        return output, p_attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation (TorchScript compatible)
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        # Ensure d_model is divisible by h
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by h ({h})")

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        
        # Define separate query, key, value projections for clarity
        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # Huggingface LoRA compatible
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)


        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Project inputs to multi-head queries, keys, values
        # query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        #     for l, x in zip(self.linear_layers, (query, key, value))]
        query = self.q_proj(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)


        # 2) Apply attention on all the projected vectors in batch
        attn_output, _ = self.attention(query, key, value, mask=mask, apply_dropout=True)

        # 3) "Concat" using a view and apply a final linear
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.output_linear(concat)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    TorchScript compatible version.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        self.a = nn.Parameter(torch.tensor(1e-8)) if enable_res_parameter else None
    
    # Separate implementations to avoid lambda functions and dynamic checks
    def forward_with_list(self, x_list: List[torch.Tensor], sublayer_output):
        """Process list input with pre-computed sublayer output"""
        if self.enable:
            return self.norm(x_list[1] + self.dropout(self.a * sublayer_output))
        else:
            return self.norm(x_list[1] + self.dropout(sublayer_output))
            
    def forward_with_tensor(self, x, sublayer_output):
        """Process tensor input with pre-computed sublayer output"""
        if self.enable:
            return self.norm(x + self.dropout(self.a * sublayer_output))
        else:
            return self.norm(x + self.dropout(sublayer_output))
            
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], sublayer_output):
        """Apply residual connection with pre-computed sublayer output"""
        if isinstance(x, list):
            return self.forward_with_list(x, sublayer_output)
        else:
            return self.forward_with_tensor(x, sublayer_output)


class PointWiseFeedForward(nn.Module):
    """
    Feed-forward network (TorchScript compatible)
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply operations in sequence for clarity
        hidden = self.linear1(x)
        activated = self.activation(hidden)
        output = self.linear2(activated)
        return self.dropout(output)


class TransformerBlock(nn.Module):
    """
    TRM layer (TorchScript compatible version)
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def compute_attention(self, x, mask: Optional[torch.Tensor] = None):
        """Compute self-attention output for input x"""
        return self.attn(x, x, x, mask=mask)
        
    def compute_ffn(self, x):
        """Apply feed-forward network to input"""
        return self.ffn(x)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Compute attention output
        attn_output = self.compute_attention(x, mask=mask)
        # Apply first skip connection with pre-computed output
        x = self.skipconnect1(x, attn_output)
        # Compute FFN output
        ffn_output = self.compute_ffn(x)
        # Apply second skip connection
        x = self.skipconnect2(x, ffn_output)
        return x