import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.utils.typing import *
from vae.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from .utils import init_linear, MLP

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        use_flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = checkpoint(self.attention, (x, data), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, use_flash: bool = False, n_data: Optional[int] = None):

        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x
    



class Lrpe(nn.Module):
    def __init__(
        self,
        num_heads=8,
        embed_dim=64,
    ):
        super().__init__()
        d = num_heads * embed_dim
        
        self.index = torch.empty(0)
        self.theta = nn.Parameter(10000**(-2 / d *
                                                torch.arange(d)).reshape(
                                                    num_heads, 1, -1))

    def forward(self, x, offset=0):
        # x: b, h, n, d
        # offset: for k, v cache
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        
        x = torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)

        return x


    

class NormLinearAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        hidden_dim,
        heads,
        dropout=0.0,
        context_dim = 0,
        bias = False,
        use_lrpe=False,
        norm_qk=False,
        **kwargs
    ):
        super().__init__()

        # breakpoint()
        
        bias = bias,
        self.n_head = heads
        self.use_lrpe = use_lrpe
        self.norm_qk = norm_qk
                
        self.q_proj = nn.Linear(query_dim, hidden_dim,  bias=bias)
        self.k_proj = nn.Linear(query_dim, hidden_dim,  bias=bias)
        self.v_proj = nn.Linear(query_dim, query_dim,  bias=bias)
        self.u_proj = nn.Linear(query_dim, query_dim,  bias=bias)
        
        self.out_proj = nn.Linear(query_dim, query_dim,  bias=bias)

        if self.use_lrpe:
            self.lrpe = Lrpe(num_heads=self.n_head, embed_dim=hidden_dim//self.n_head)
            
        self.act = F.relu
        self.norm = nn.LayerNorm(query_dim)

        self.clip = True
        self.eps = 1e-5

    def abs_clamp(self, t):
        min_mag = 1e-2
        max_mag = 100
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag)*sign
    
    def forward(
        self,
        query,
        context = None
    ):
        
        # linear map
        
        q = self.q_proj(query)
        u = self.u_proj(query)
        if context is not None:
            k = self.k_proj(context)
            v = self.v_proj(context)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)
        
        if self.norm_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        # breakpoint()
        
        kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
        if self.clip:
            kv = self.abs_clamp(kv)
        output = torch.einsum('... n d, ... d e -> ... n e', q, kv)


        # breakpoint()
        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)
        
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output

    def forward_left(
        self,
        query,
        context = None
    ):

        q = self.q_proj(query)
        u = self.u_proj(query)
        
        if context is not None:
            k = self.k_proj(context)
            v = self.v_proj(context)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
            
        # reshape
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)

        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)

        # lrpe
        if self.use_lrpe:
            offset = 0
            q = self.lrpe(q, offset=offset)
            k = self.lrpe(k, offset=offset)

        qk = torch.einsum("... m d, ... n d -> ... m n", q, k)
        # if self.clip:
            # qk = self.abs_clamp(qk)

        output = torch.einsum('... m n, ... n e -> ... m e', qk, v)
        
        # reshape
        output = rearrange(output, 'b h n d -> b n (h d)')
        # normalize
        output = self.norm(output)
        
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output






class BasicAttentionBlock(nn.Module):
    def __init__(
        self,
        query_dim,
        hidden_dim,
        heads,
        dropout=0.0,
        context_dim = 0,
        bias = False,
        norm_qk=False,
        **kwargs
    ):
        super().__init__()

        # breakpoint()
        
        bias = bias,
        self.n_head = heads
        
        self.out_proj = nn.Linear(hidden_dim, query_dim,  bias=bias)
        
        self.q_proj = nn.Linear(query_dim, hidden_dim,  bias=bias)
        self.k_proj = nn.Linear(query_dim, hidden_dim,  bias=bias)
        self.v_proj = nn.Linear(query_dim, hidden_dim,  bias=bias)

        self.norm_qk = norm_qk
        
        self.clip = True
        self.eps = 1e-5
    
    def forward(self,
                query,
                context = None):
        
        q = self.q_proj(query)
        if context is not None:
            k = self.k_proj(context)
            v = self.v_proj(context)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        
        q, k, v = map(
            lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_head),
            [q, k, v])
        
        if self.norm_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        
        # print(q.shape, k.shape, v.shape)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = output.to(query.dtype)
        
        output = self.out_proj(output)
        
        return output
    
    