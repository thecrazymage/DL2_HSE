import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Attention implementations
# ---------------------------

def mha_looped(q, k, v, n_heads):
    """
    q,k,v: [B, N, D]
    returns: [B, N, D]
    """
    B, N, D = q.shape
    assert D % n_heads == 0, "D must be divisible by n_heads"
    d_head = D // n_heads
    scale = 1.0 / d_head ** 0.5

    # Split last dim into heads: list of [B, N, d_head]
    qs = q.split(d_head, -1)  # method for splitting the tensor across the ast dimension with specified slice size (`d_head`)
    ks = k.split(d_head, -1)
    vs = v.split(d_head, -1)

    head_outputs = []
    for qi, ki, vi in zip(qs, ks, vs):
        # [B, N, d_head] @ [B, d_head, N] -> [B, N, N]
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        attn = scores.softmax(-1)
        # [B, N, N] @ [B, N, d_head] -> [B, N, d_head]
        out_i = torch.matmul(attn, vi)
        head_outputs.append(out_i)

    # Concatenate heads back: [B, N, D]
    out = torch.cat(head_outputs, dim=-1)
    return out


def mha_batched(q, k, v, n_heads):
    """
    q,k,v: [B, N, D]
    returns: [B, N, D]
    """
    B, N, D = q.shape
    assert D % n_heads == 0, "D must be divisible by n_heads"
    d_head = D // n_heads
    scale = 1.0 / d_head ** 0.5

    # Reshape to [B, n_heads, N, d_head]
    qh = q.view(B, N, n_heads, d_head).transpose(1, 2)  # [B, H, N, d_head]
    kh = k.view(B, N, n_heads, d_head).transpose(1, 2)  # [B, H, N, d_head]
    vh = v.view(B, N, n_heads, d_head).transpose(1, 2)  # [B, H, N, d_head]

    # Attention scores: [B, H, N, N]
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    attn = scores.softmax(dim=-1)

    # Weighted sum: [B, H, N, d_head]
    out_heads = torch.matmul(attn, vh)

    # Merge heads back to [B, N, D]
    out = out_heads.transpose(1, 2).contiguous().view(B, N, D)
    return out


def mha_sdpa(q, k, v, n_heads):
    """
    Reference using torch.nn.functional.scaled_dot_product_attention
    q,k,v: [B, N, D] -> internally reshaped to [B,H,N,d]
    returns: [B, N, D]
    """
    B, N, D = q.shape
    d_head = D // n_heads
    # [B,H,N,d]
    qh = q.view(B, N, n_heads, d_head).transpose(1, 2)
    kh = k.view(B, N, n_heads, d_head).transpose(1, 2)
    vh = v.view(B, N, n_heads, d_head).transpose(1, 2)
    # Let SDPA choose the scale (it uses 1/sqrt(d_head) if scale=None)
    oh = F.scaled_dot_product_attention(qh, kh, vh,
                                        attn_mask=None,
                                        dropout_p=0.0,
                                        is_causal=False,
                                        scale=None)           # [B,H,N,d]
    return oh.transpose(1, 2).contiguous().view(B, N, D)      # [B,N,D]


# ---------------------------
# Model: tiny ViT-like encoder block
# ---------------------------

class SelectableMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, backend: str):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.backend = backend.lower()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.backend == "sdpa":
            out = mha_sdpa(q, k, v, self.n_heads)
        elif self.backend == "looped":
            out = mha_looped(q, k, v, self.n_heads)
        elif self.backend == "batched":
            out = mha_batched(q, k, v, self.n_heads)
        else:
            raise ValueError(f"Unknown attention backend: {self.backend}")
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, backend: str, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelectableMHA(d_model, n_heads, backend)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, N, D]
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class TinyViT(nn.Module):
    """
    Very small ViT-ish model for 32x32 images, patch_size=4 -> 8x8=64 tokens.
    """
    def __init__(self, num_classes=10, img_size=32, patch=4, d_model=256, depth=4, n_heads=8, mlp_ratio=4.0, backend="sdpa"):
        super().__init__()
        assert img_size % patch == 0
        self.num_patches = (img_size // patch) * (img_size // patch)  # 64
        self.patch = patch
        in_ch = 3
        self.patch_embed = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)  # [B, D, H/ps, W/ps]
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, backend=backend) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        # x: [B,3,32,32]
        x = self.patch_embed(x)              # [B,D,8,8]
        x = x.flatten(2).transpose(1, 2)     # [B,64,D]
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                    # global average over tokens
        return self.head(x)

