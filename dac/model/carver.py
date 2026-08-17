import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from audiotools.ml import BaseModel

from .dac import Encoder, ResidualUnit, init_weights
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d


# --- decoder ---

class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, dilations=(1, 3, 9)):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2), output_padding=stride % 2,
            ),
        ]
        for d in dilations:
            layers.append(ResidualUnit(output_dim, dilation=d))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class NonCausalDecoder(nn.Module):
    def __init__(self, input_channel: int, channels: int, rates, d_out: int = 1, dilations=(1, 3, 9)):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            in_dim = channels // 2 ** i
            out_dim = channels // 2 ** (i + 1)
            layers.append(_DecoderBlock(in_dim, out_dim, stride, dilations=dilations))
        out_dim = channels // 2 ** len(rates)
        layers += [
            Snake1d(out_dim),
            WNConv1d(out_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --- transformer ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000.0) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1).float()


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    return x_out.flatten(3).type_as(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, head_dim: int, ffn_dim: int, dropout: float = 0.0,
                 layer_scale_init: float = 1e-2):
        super().__init__()
        assert n_head * head_dim == dim, f"n_head*head_dim ({n_head}*{head_dim}) must == dim ({dim})"
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.attn_layer_scale = LayerScale(dim, layer_scale_init)
        self.ffn_layer_scale = LayerScale(dim, layer_scale_init)

    def _attn(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, d]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        x = x + self.attn_layer_scale(self._attn(self.attention_norm(x), freqs_cis))
        h = self.ffn_norm(x)
        x = x + self.ffn_layer_scale(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x


class TransformerStack(nn.Module):
    """A stack of bidirectional TransformerBlocks. I/O is [B, C, T]."""

    def __init__(self, n_layers: int, dim: int, n_head: int, head_dim: int, ffn_dim: int, c_in: int,
                 dropout: float = 0.0, use_temporal_embed: bool = True, max_frames: int = 4096,
                 init_std: float = 0.02, layer_scale_init: float = 1e-2):
        super().__init__()
        self.dim = dim
        self.use_temporal_embed = use_temporal_embed
        if dim != c_in:
            self.proj_in = nn.Linear(c_in, dim, bias=False)
            self.proj_out = nn.Linear(dim, c_in, bias=False)
            nn.init.normal_(self.proj_in.weight, std=init_std)
            nn.init.normal_(self.proj_out.weight, std=init_std)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        if use_temporal_embed:
            self.temporal_embed = nn.Parameter(torch.randn(max_frames, dim) * init_std)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_head, head_dim, ffn_dim, dropout, layer_scale_init)
            for _ in range(n_layers)
        ])

    def forward(self, h: Tensor, freqs_cis: Tensor) -> Tensor:
        T = h.shape[-1]
        assert T <= freqs_cis.shape[0], (
            f"sequence length T={T} exceeds max_frames={freqs_cis.shape[0]}; increase max_frames."
        )
        x = self.proj_in(h.transpose(1, 2))            # [B, T, dim]
        if self.use_temporal_embed:
            x = x + self.temporal_embed[:T].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, freqs_cis[:T])
        return self.proj_out(x).transpose(1, 2)        # [B, C, T]


# --- model ---

class Carver(BaseModel):
    def __init__(
        self,
        # --- backbone ---
        encoder_dim: int = 64,
        encoder_rates: List[int] = [3, 5, 8, 8],
        latent_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 5, 3],
        decoder_dilations: List[int] = [1, 3, 9],
        sample_rate: int = 24000,
        logvar_clamp: float = 20.0,
        # --- transformer ---
        transformer_dim: int = 1024,
        n_enc_transformer_layers: int = 4,
        n_pre_layers: int = 8,
        n_post_layers: int = 8,
        n_head: int = 16,
        transformer_ffn: int = 3072,
        transformer_dropout: float = 0.0,
        layer_scale_init: float = 1e-2,
        rope_base: float = 10000.0,
        max_frames: int = 4096,
        use_temporal_embed: bool = True,
        init_std: float = 0.02,
        # --- routing ---
        min_rate: float = 0.0625,
        **_unused,  # tolerate training-only kwargs stored in metadata
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp
        self.min_rate = min_rate
        self.hop_length = math.prod(encoder_rates)
        self.z_channels = encoder_dim * (2 ** len(encoder_rates))

        # Encoder: reuse dac.Encoder, drop its final Snake+conv, attach mu/logvar heads.
        self.encoder = Encoder(d_model=encoder_dim, strides=encoder_rates, d_latent=latent_dim)
        self.encoder.block = self.encoder.block[:-2]
        self.fc_mu = nn.Sequential(
            Snake1d(self.z_channels), WNConv1d(self.z_channels, latent_dim, kernel_size=3, padding=1)
        )
        self.fc_logvar = nn.Sequential(
            Snake1d(self.z_channels), WNConv1d(self.z_channels, latent_dim, kernel_size=3, padding=1)
        )
        self.post_quant_conv = WNConv1d(latent_dim, self.z_channels, kernel_size=3, padding=1)

        self.decoder = NonCausalDecoder(
            input_channel=self.z_channels, channels=decoder_dim,
            rates=decoder_rates, d_out=1, dilations=decoder_dilations,
        )

        # Three Transformer stacks: enc_transformer / pre_module / post_module.
        head_dim = transformer_dim // n_head
        self.head_dim = head_dim
        stack_kwargs = dict(
            dim=transformer_dim, n_head=n_head, head_dim=head_dim, ffn_dim=transformer_ffn,
            c_in=self.z_channels, dropout=transformer_dropout, use_temporal_embed=use_temporal_embed,
            max_frames=max_frames, init_std=init_std, layer_scale_init=layer_scale_init,
        )
        self.enc_transformer = TransformerStack(n_enc_transformer_layers, **stack_kwargs) \
            if n_enc_transformer_layers > 0 else None
        self.pre_module = TransformerStack(n_pre_layers, **stack_kwargs)
        self.post_module = TransformerStack(n_post_layers, **stack_kwargs)
        self.register_buffer("freqs_cis", precompute_freqs_cis(max_frames, head_dim, rope_base),
                             persistent=False)

        self.apply(init_weights)

    # --- encode / decode ---
    def preprocess(self, audio_data: Tensor) -> Tensor:
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return F.pad(audio_data, (0, right_pad))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = torch.clamp(logvar, -self.logvar_clamp, self.logvar_clamp)
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, audio_data: Tensor) -> Tensor:
        h = self.encoder.block(self.preprocess(audio_data))    # [B, z_channels, T]
        if self.enc_transformer is not None:
            h = self.enc_transformer(h, self.freqs_cis)
        h = self.pre_module(h, self.freqs_cis)
        return self.reparameterize(self.fc_mu(h), self.fc_logvar(h))

    def decode(self, z: Tensor) -> Tensor:
        h = self.post_quant_conv(z)                            # [B, z_channels, T]
        h = self.post_module(h, self.freqs_cis)
        return self.decoder(h)

    # --- routing: per-frame waveform L1 error ---
    def _per_frame_loss(self, audio: Tensor, recon: Tensor, T: int) -> Tensor:
        L = min(audio.shape[-1], recon.shape[-1])
        err = (audio[..., :L] - recon[..., :L]).abs().squeeze(1)
        usable = (L // self.hop_length) * self.hop_length
        err = err[:, :usable].reshape(err.shape[0], -1, self.hop_length).mean(-1)
        if err.shape[1] != T:
            err = F.interpolate(err.unsqueeze(1), size=T, mode="linear", align_corners=False).squeeze(1)
        return err[:, :T]

    def forward(self, audio_data: Tensor, manual_rate: float = 1.0):
        """Reconstruct at a fixed keep-ratio ``manual_rate`` in (0, 1].

        Frames are scored by their waveform L1 reconstruction error and the top
        ``round(manual_rate * T)`` are kept; the rest are zeroed before decoding.
        """
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(1)
        audio_data = audio_data.to(torch.float32)
        length = audio_data.shape[-1]

        z = self.encode(audio_data)
        T = z.shape[-1]

        recon_full = self.decode(z)
        chunk_loss = self._per_frame_loss(audio_data, recon_full, T)

        k = max(1, min(T, int(round(float(manual_rate) * T))))
        order = torch.argsort(chunk_loss, dim=-1, descending=True)
        ranks = torch.argsort(order, dim=-1)
        keep = ranks < k                                       # [B, T] top-k by L1 error

        z_masked = z * keep.unsqueeze(1).to(z.dtype)
        recon = self.decode(z_masked)

        return {
            "audio": recon[..., :length],
            "keep_mask": keep,
            "mask_rate": keep.float().mean(),
        }
