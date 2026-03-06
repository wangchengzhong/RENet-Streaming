import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import GRU, Linear, LayerNorm, Dropout

import math
import numpy as np


class ComplexConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConv1d, self).__init__()
        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.fused_conv = None

    def fuse_weights(self):
        w_r = self.conv_re.weight
        w_i = self.conv_im.weight

        row_real = torch.cat([w_r, -w_i], dim=1)
        row_imag = torch.cat([w_i, w_r], dim=1)
        w_fused = torch.cat([row_real, row_imag], dim=0)

        fused = nn.Conv1d(
            in_channels=self.conv_re.in_channels * 2,
            out_channels=self.conv_re.out_channels * 2,
            kernel_size=self.conv_re.kernel_size,
            stride=self.conv_re.stride,
            padding=self.conv_re.padding,
            dilation=self.conv_re.dilation,
            groups=self.conv_re.groups,
            bias=False,
        )
        fused.weight.data.copy_(w_fused)
        self.fused_conv = fused.to(device=w_r.device, dtype=w_r.dtype)

    def clear_fused(self):
        self.fused_conv = None

    def forward(self, real, imag):
        if self.fused_conv is not None and not self.training:
            x_cat = torch.cat([real, imag], dim=1)
            out_cat = self.fused_conv(x_cat)
            return torch.chunk(out_cat, 2, dim=1)
            
        real_conv_real = self.conv_re(real)
        real_conv_imag = self.conv_re(imag)
        imag_conv_real = self.conv_im(real)
        imag_conv_imag = self.conv_im(imag)

        real_ = real_conv_real - imag_conv_imag
        imaginary_ = real_conv_imag + imag_conv_real
        
        return real_, imaginary_

class ComplexConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConvTranspose1d, self).__init__()
        def _time_pair(value, default_first):
            if isinstance(value, tuple):
                if len(value) == 2:
                    return value
                if len(value) == 1:
                    return (default_first, value[0])
            return (default_first, value)

        k = kernel_size[-1] if isinstance(kernel_size, tuple) else kernel_size
        stride = _time_pair(kwargs.pop("stride", 1), 1)
        padding = _time_pair(kwargs.pop("padding", 0), 0)
        dilation = _time_pair(kwargs.pop("dilation", 1), 1)
        output_padding = _time_pair(kwargs.pop("output_padding", 0), 0)
        groups = kwargs.pop("groups", 1)

        self.conv_re = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=(1, k),
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            bias=False,
        )
        self.conv_im = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=(1, k),
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            bias=False,
        )

    def forward(self, real, imag):
        real_2d = real.unsqueeze(-2)
        imag_2d = imag.unsqueeze(-2)

        real_conv_real = self.conv_re(real_2d)
        real_conv_imag = self.conv_re(imag_2d)
        imag_conv_real = self.conv_im(real_2d)
        imag_conv_imag = self.conv_im(imag_2d)

        real_ = (real_conv_real - imag_conv_imag).squeeze(-2)
        imaginary_ = (real_conv_imag + imag_conv_real).squeeze(-2)
        
        return real_, imaginary_

class ComplexRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-10):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x_r, x_i) -> tuple[torch.Tensor, torch.Tensor]:
        mag_sq = x_r**2 + x_i**2 

        mean_mag_sq = mag_sq.mean(dim=-1, keepdim=True)

        inv_rms = torch.rsqrt(mean_mag_sq + self.eps)

        gamma = self.gamma.view(1, 1, -1)
        
        # (B, L, C) * (B, L, 1) * (1, 1, C)
        x_r = x_r * inv_rms * gamma
        x_i = x_i * inv_rms * gamma

        return x_r, x_i

class ComplexFFN(nn.Module):
    def __init__(self, chn=16, chn_inner=64, guide_chn=48, conv1d_kernel=4, conv1d_shift=1, dropout=0., causal=False, **kwargs):
        super().__init__()

        self.chn = chn
        self.chn_inner = chn_inner
        self.guide_chn = guide_chn
        self.conv1d = ComplexConv1d(chn, chn_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.gate_act = nn.SiLU()
        self.gate_ln = LayerNorm(chn_inner)
        self.deconv1d = ComplexConvTranspose1d(chn_inner, chn, conv1d_kernel, stride=conv1d_shift) if not causal else None
        self.causal_conv_out = ComplexConv1d(chn_inner, chn, conv1d_kernel, stride=conv1d_shift) if causal else None
        self.dropout = nn.Dropout(dropout)
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift
        self.causal = causal
        if self.causal and self.conv1d_shift != 1:
            raise ValueError('Causal ComplexFFN currently supports conv1d_shift=1 only.')

    def fuse_inference_kernels(self):
        self.conv1d.fuse_weights()
        if self.causal_conv_out is not None:
            self.causal_conv_out.fuse_weights()

    def forward(self, x_real_orig, x_imag_orig):
        """ forward
        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s2, h = x_real_orig.shape
        # x = x.contiguous().view(b * s1, s2, h)
        x_real = x_real_orig.transpose(-1, -2)
        x_imag = x_imag_orig.transpose(-1, -2)
        # padding
        if self.causal:
            left_pad = self.diff_ks
            right_pad = 0
            crop_start = 0
        else:
            seq_len = (
                math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
                + self.conv1d_kernel
            )
            left_pad = self.diff_ks
            right_pad = seq_len - s2 - self.diff_ks
            crop_start = self.diff_ks

        x_real = F.pad(x_real, (left_pad, right_pad))
        x_imag = F.pad(x_imag, (left_pad, right_pad))
        # conv-deconv1d
        x_real, x_imag = self.conv1d(x_real, x_imag)
        x_act = torch.sqrt(x_real[..., self.chn_inner:, :] ** 2 + x_imag[..., self.chn_inner:, :] ** 2 + 1e-9)
        x_act = self.gate_ln(x_act.permute(0,2,1)).permute(0,2,1)
        gate = self.gate_act(x_act)
        x_real = x_real[..., : self.chn_inner, :] * gate
        x_imag = x_imag[..., : self.chn_inner, :] * gate

        if self.causal:
            x_real = F.pad(x_real, (self.diff_ks, 0))
            x_imag = F.pad(x_imag, (self.diff_ks, 0))
            x_real, x_imag = self.causal_conv_out(x_real, x_imag)
        else:
            x_real, x_imag = self.deconv1d(x_real, x_imag)
        x_real = x_real.transpose(-1, -2)
        x_imag = x_imag.transpose(-1, -2)
        # cut necessary part
        x_real = x_real[..., crop_start : crop_start + s2, :]
        x_imag = x_imag[..., crop_start : crop_start + s2, :]
        return x_real, x_imag

    def forward_stream(self, x_real_t, x_imag_t, in_cache_r, in_cache_i, mid_cache_r, mid_cache_i):
        if not self.causal:
            raise ValueError('forward_stream is only supported for causal ComplexFFN.')
        if self.conv1d_shift != 1:
            raise ValueError('forward_stream currently supports conv1d_shift=1 only.')

        x_real = x_real_t.transpose(-1, -2)
        x_imag = x_imag_t.transpose(-1, -2)

        if in_cache_r.shape[-1] == self.conv1d_kernel:
            conv_in_r = torch.roll(in_cache_r, shifts=-1, dims=-1)
            conv_in_i = torch.roll(in_cache_i, shifts=-1, dims=-1)
            conv_in_r[..., -1:] = x_real
            conv_in_i[..., -1:] = x_imag
            in_cache_r_out = conv_in_r
            in_cache_i_out = conv_in_i
        else:
            conv_in_r = torch.empty(
                in_cache_r.size(0),
                in_cache_r.size(1),
                self.diff_ks + 1,
                device=in_cache_r.device,
                dtype=in_cache_r.dtype,
            )
            conv_in_i = torch.empty(
                in_cache_i.size(0),
                in_cache_i.size(1),
                self.diff_ks + 1,
                device=in_cache_i.device,
                dtype=in_cache_i.dtype,
            )
            conv_in_r[..., :-1] = in_cache_r
            conv_in_i[..., :-1] = in_cache_i
            conv_in_r[..., -1:] = x_real
            conv_in_i[..., -1:] = x_imag
            in_cache_r_out = conv_in_r[..., -self.diff_ks:]
            in_cache_i_out = conv_in_i[..., -self.diff_ks:]

        x_real, x_imag = self.conv1d(conv_in_r, conv_in_i)
        x_act = torch.sqrt(x_real[..., self.chn_inner:, :] ** 2 + x_imag[..., self.chn_inner:, :] ** 2 + 1e-9)
        x_act = self.gate_ln(x_act.permute(0, 2, 1)).permute(0, 2, 1)
        gate = self.gate_act(x_act)
        x_real = x_real[..., : self.chn_inner, :] * gate
        x_imag = x_imag[..., : self.chn_inner, :] * gate

        if mid_cache_r.shape[-1] == self.conv1d_kernel:
            conv_mid_r = torch.roll(mid_cache_r, shifts=-1, dims=-1)
            conv_mid_i = torch.roll(mid_cache_i, shifts=-1, dims=-1)
            conv_mid_r[..., -1:] = x_real
            conv_mid_i[..., -1:] = x_imag
            mid_cache_r_out = conv_mid_r
            mid_cache_i_out = conv_mid_i
        else:
            conv_mid_r = torch.empty(
                mid_cache_r.size(0),
                mid_cache_r.size(1),
                self.diff_ks + 1,
                device=mid_cache_r.device,
                dtype=mid_cache_r.dtype,
            )
            conv_mid_i = torch.empty(
                mid_cache_i.size(0),
                mid_cache_i.size(1),
                self.diff_ks + 1,
                device=mid_cache_i.device,
                dtype=mid_cache_i.dtype,
            )
            conv_mid_r[..., :-1] = mid_cache_r
            conv_mid_i[..., :-1] = mid_cache_i
            conv_mid_r[..., -1:] = x_real
            conv_mid_i[..., -1:] = x_imag
            mid_cache_r_out = conv_mid_r[..., -self.diff_ks:]
            mid_cache_i_out = conv_mid_i[..., -self.diff_ks:]
        x_real, x_imag = self.causal_conv_out(conv_mid_r, conv_mid_i)

        x_real = x_real.transpose(-1, -2)
        x_imag = x_imag.transpose(-1, -2)

        return x_real, x_imag, in_cache_r_out, in_cache_i_out, mid_cache_r_out, mid_cache_i_out

class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0.1):
        super(FFN, self).__init__()
        if bidirectional:
            self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional, batch_first=True)
            self.linear = Linear(d_model*2*2, d_model)

        else:
            self.gru = GRU(d_model, d_model*2, 2, bidirectional=bidirectional, batch_first=True)
            self.linear = Linear(d_model*2, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        if self.training:
            self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

    def forward_stream(self, x_t, h_cache):
        if self.gru.bidirectional:
            raise ValueError('forward_stream supports unidirectional GRU only.')
        if self.training:
            self.gru.flatten_parameters()
        x, h_out = self.gru(x_t, h_cache)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x, h_out


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_i = nn.Parameter(torch.empty(out_features, in_features))

        self.use_bias = bias
        if self.use_bias:
            self.bias_r = nn.Parameter(torch.empty(out_features))
            self.bias_i = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        # Initialize weights similar to nn.Linear for stable training
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_r, -bound, bound)
            nn.init.uniform_(self.bias_i, -bound, bound)

    def forward(self, x_r, x_i):
  
        out_r = F.linear(x_r, self.weight_r) - F.linear(x_i, self.weight_i)
        out_i = F.linear(x_r, self.weight_i) + F.linear(x_i, self.weight_r)

        if self.use_bias:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i

        return out_r, out_i

class CustomAttention(nn.Module):
    def __init__(self, amp_dim=48, ang_dim=16, num_heads=4, amp_qk_head_dim=12, ang_qk_head_dim=6, amp_v_head_dim=12, ang_v_head_dim=6, causal=False, attn_lookback=100):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        self.attn_lookback = attn_lookback
        
        # Dimensions per head
        self.amp_qk_head_dim = amp_qk_head_dim
        self.ang_qk_head_dim = ang_qk_head_dim
        self.amp_v_head_dim = amp_v_head_dim
        self.ang_v_head_dim = ang_v_head_dim
        
        # Total internal dimensions (Head_Dim * Num_Heads)
        self.inner_dim_qk_amp = num_heads * amp_qk_head_dim
        self.inner_dim_qk_ang = num_heads * ang_qk_head_dim
        self.inner_dim_v_amp = num_heads * amp_v_head_dim
        self.inner_dim_v_ang = num_heads * ang_v_head_dim

        # Projections
        # Q and K
        self.to_q_amp = nn.Linear(amp_dim, self.inner_dim_qk_amp, bias=False)
        self.to_k_amp = nn.Linear(amp_dim, self.inner_dim_qk_amp, bias=False)
        self.to_q_ang = ComplexLinear(ang_dim, self.inner_dim_qk_ang, bias=False)
        self.to_k_ang = ComplexLinear(ang_dim, self.inner_dim_qk_ang, bias=False)
        # V (Values)
        self.to_v_amp = nn.Linear(amp_dim, self.inner_dim_v_amp, bias=False)
        self.to_v_ang = ComplexLinear(ang_dim, self.inner_dim_v_ang, bias=False)
        
        # Output projections
        self.to_out_amp = nn.Linear(self.inner_dim_v_amp, amp_dim)
        self.to_out_ang = ComplexLinear(self.inner_dim_v_ang, ang_dim, bias=False)

    def forward(self, x_ang, x_amp):
        B, L, CA = x_ang.shape
        ang_r, ang_i = torch.split(x_ang, [CA//2, CA//2], dim=-1)

        # ==========================================================
        # 1. Project Q, K, V
        # ==========================================================
        
        # --- Query & Key ---
        q_amp = self.to_q_amp(x_amp)      # [B, L, H * D_qk_amp]
        k_amp = self.to_k_amp(x_amp)
        q_ang_r, q_ang_i = self.to_q_ang(ang_r, ang_i) # [B, L, H * D_qk_ang]
        k_ang_r, k_ang_i = self.to_k_ang(ang_r, ang_i)

        # Reshape and Transpose Q/K
        # Function to reshape: (B, L, H*D) -> (B, H, L, D)
        def reshape_head(x, head_dim):
            return x.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        q_amp = reshape_head(q_amp, self.amp_qk_head_dim)
        k_amp = reshape_head(k_amp, self.amp_qk_head_dim)
        q_ang_r = reshape_head(q_ang_r, self.ang_qk_head_dim)
        q_ang_i = reshape_head(q_ang_i, self.ang_qk_head_dim)
        k_ang_r = reshape_head(k_ang_r, self.ang_qk_head_dim)
        k_ang_i = reshape_head(k_ang_i, self.ang_qk_head_dim)
        # Concatenate parts for Q and K
        # Structure: [Real_Angle, Imag_Angle, Amplitude]
        q = torch.cat((q_ang_r, q_ang_i, q_amp), dim=-1)
        k = torch.cat((k_ang_r, k_ang_i, k_amp), dim=-1)

        # --- Value (V) ---
        v_amp = self.to_v_amp(x_amp)
        v_ang_r, v_ang_i = self.to_v_ang(ang_r, ang_i)

        # Reshape V to (B, H, L, D)
        v_amp = reshape_head(v_amp, self.amp_v_head_dim)
        v_ang_r = reshape_head(v_ang_r, self.ang_v_head_dim)
        v_ang_i = reshape_head(v_ang_i, self.ang_v_head_dim)

        # Concatenate V: [Amplitude, Real_Angle, Imag_Angle]
        v_combined = torch.cat((v_ang_r, v_ang_i, v_amp), dim=-1)

        # ==========================================================
        # 2. Attention
        # ==========================================================
        # out shape: [B, H, L, Total_V_Head_Dim]

        attn_mask = None
        if self.causal:
            q_idx = torch.arange(L, device=q.device).unsqueeze(1)
            k_idx = torch.arange(L, device=q.device).unsqueeze(0)
            lower_bound = q_idx - (self.attn_lookback - 1)
            valid = (k_idx <= q_idx) & (k_idx >= lower_bound)
            min_value = torch.finfo(q.dtype).min
            attn_mask = torch.full((L, L), min_value, device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(valid, 0.0)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v_combined, attn_mask=attn_mask)
        else:
            scale = 1.0 / math.sqrt(q.size(-1))
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v_combined)

        # ==========================================================
        # 3.  Output Processing
        # ==========================================================
        
        split_sizes = [self.ang_v_head_dim, self.ang_v_head_dim, self.amp_v_head_dim]
        out_ang_r, out_ang_i, out_amp = torch.split(out, split_sizes, dim=-1)

        # Function to merge heads: (B, H, L, D) -> (B, L, H*D)
        def merge_head(x):
            return x.transpose(1, 2).contiguous().view(B, L, -1)

        # Merge heads individually for each component

        out_amp = merge_head(out_amp)      # [B, L, H * amp_v_dim]
        out_ang_r = merge_head(out_ang_r)  # [B, L, H * ang_v_dim]
        out_ang_i = merge_head(out_ang_i)  # [B, L, H * ang_v_dim]

        # ==========================================================
        # 4. Final Projection
        # ==========================================================

        out_amp = self.to_out_amp(out_amp)
        out_ang_r, out_ang_i = self.to_out_ang(out_ang_r, out_ang_i)

        return out_ang_r, out_ang_i, out_amp

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(RMSNorm, self).__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        # Create a learnable parameter for the gain (gamma), initialized to 1
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Calculate the Root Mean Square (RMS) along the last dimension
        norm = torch.mean(x ** 2, dim=-1, keepdim=True)
        
        # 2. Normalize x by dividing by RMS (adding eps for stability)
        x_normed = x * torch.rsqrt(norm + self.eps)

        # 3. Apply the learnable gain (affine transformation)
        return self.g * x_normed
    
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        h,
        causal=False,
        attn_lookback=100,
        dropout=0.0
    ):
        super().__init__()
        amp_chn = h.amp_chn
        ang_chn = h.ang_chn
        self.amp_chn = amp_chn
        self.ang_chn = ang_chn
        self.norm1 = RMSNorm(amp_chn)
        self.cnorm1 = ComplexRMSNorm(ang_chn)

        self.att = CustomAttention(
            amp_chn,
            ang_chn,
            h.n_heads,
            h.amp_attnhead_dim,
            h.ang_attnhead_dim,
            h.amp_attnhead_dim,
            h.ang_attnhead_dim,
            causal=causal,
            attn_lookback=attn_lookback,
        )

        self.norm2 = RMSNorm(amp_chn)
        self.cnorm2 = ComplexRMSNorm(ang_chn)

        self.amp_ffn = FFN(amp_chn, bidirectional=not causal, dropout=dropout)
        self.ang_ffn = ComplexFFN(ang_chn, dropout=dropout, causal=causal)

        self.norm3 = RMSNorm(amp_chn)
        self.cnorm3 = ComplexRMSNorm(ang_chn)
        self.register_buffer('_attn_idx_cache', torch.empty(0), persistent=False)
        self.attn_scale = 1.0 / math.sqrt(self.att.ang_qk_head_dim * 2 + self.att.amp_qk_head_dim)

    def fuse_inference_kernels(self):
        self.ang_ffn.fuse_inference_kernels()

    def forward(self, x):
        x_ang_r, x_ang_i, x_amp = torch.split(x, [self.ang_chn, self.ang_chn, self.amp_chn], dim=-1)

        x_amp_t = self.norm1(x_amp)
        x_ang_r_t, x_ang_i_t = self.cnorm1(x_ang_r, x_ang_i)
        x_ang_r_t, x_ang_i_t, x_amp_t = self.att(torch.cat((x_ang_r_t, x_ang_i_t), dim=-1), x_amp_t)

        x_amp, x_ang_r, x_ang_i = x_amp + x_amp_t, x_ang_r + x_ang_r_t, x_ang_i + x_ang_i_t

        x_amp_t = self.norm2(x_amp)
        x_amp_t = self.amp_ffn(x_amp_t)
        x_amp = x_amp + x_amp_t
        x_amp = self.norm3(x_amp)

        x_ang_r_t, x_ang_i_t = self.cnorm2(x_ang_r, x_ang_i)
        x_ang_r_t, x_ang_i_t = self.ang_ffn(x_ang_r_t, x_ang_i_t)
        x_ang_r, x_ang_i = x_ang_r + x_ang_r_t, x_ang_i + x_ang_i_t
        x_ang_r, x_ang_i = self.cnorm3(x_ang_r, x_ang_i)

        return torch.cat((x_ang_r, x_ang_i, x_amp), dim=-1)

    def forward_stream(
        self,
        x_t,
        attn_k_cache,
        attn_v_cache,
        attn_kv_cache,
        attn_count,
        amp_gru_cache,
        ang_in_cache_r,
        ang_in_cache_i,
        ang_mid_cache_r,
        ang_mid_cache_i,
    ):
        x_ang_r, x_ang_i, x_amp = torch.split(x_t, [self.ang_chn, self.ang_chn, self.amp_chn], dim=-1)

        x_amp_n = self.norm1(x_amp)
        x_ang_r_n, x_ang_i_n = self.cnorm1(x_ang_r, x_ang_i)

        q_amp = self.att.to_q_amp(x_amp_n)
        q_ang_r, q_ang_i = self.att.to_q_ang(x_ang_r_n, x_ang_i_n)

        k_amp_t = self.att.to_k_amp(x_amp_n)
        k_ang_r_t, k_ang_i_t = self.att.to_k_ang(x_ang_r_n, x_ang_i_n)

        v_amp_t = self.att.to_v_amp(x_amp_n)
        v_ang_r_t, v_ang_i_t = self.att.to_v_ang(x_ang_r_n, x_ang_i_n)

        B = x_t.shape[0]
        cache_len = attn_k_cache.shape[2]

        def reshape_q(x, head_dim):
            return x.view(B, 1, self.att.num_heads, head_dim).transpose(1, 2)

        def reshape_kv_t(x, head_dim):
            return x.view(B, 1, self.att.num_heads, head_dim).transpose(1, 2)

        q_amp = reshape_q(q_amp, self.att.amp_qk_head_dim)
        q_ang_r = reshape_q(q_ang_r, self.att.ang_qk_head_dim)
        q_ang_i = reshape_q(q_ang_i, self.att.ang_qk_head_dim)
        q = torch.cat((q_ang_r, q_ang_i, q_amp), dim=-1)

        k_amp_t = reshape_kv_t(k_amp_t, self.att.amp_qk_head_dim)
        k_ang_r_t = reshape_kv_t(k_ang_r_t, self.att.ang_qk_head_dim)
        k_ang_i_t = reshape_kv_t(k_ang_i_t, self.att.ang_qk_head_dim)
        k_t = torch.cat((k_ang_r_t, k_ang_i_t, k_amp_t), dim=-1)

        v_amp_t = reshape_kv_t(v_amp_t, self.att.amp_v_head_dim)
        v_ang_r_t = reshape_kv_t(v_ang_r_t, self.att.ang_v_head_dim)
        v_ang_i_t = reshape_kv_t(v_ang_i_t, self.att.ang_v_head_dim)
        v_t = torch.cat((v_ang_r_t, v_ang_i_t, v_amp_t), dim=-1)

        scale = self.attn_scale
        if cache_len > 0:
            attn_scores_hist = torch.matmul(q, attn_k_cache.transpose(-2, -1)) * scale
            attn_scores_cur = torch.matmul(q, k_t.transpose(-2, -1)) * scale
            attn_scores = torch.cat((attn_scores_hist, attn_scores_cur), dim=-1)
        else:
            attn_scores = torch.matmul(q, k_t.transpose(-2, -1)) * scale

        if (
            self._attn_idx_cache.numel() != cache_len
            or self._attn_idx_cache.device != x_t.device
            or self._attn_idx_cache.dtype != attn_count.dtype
        ):
            self._attn_idx_cache = torch.arange(cache_len, device=x_t.device, dtype=attn_count.dtype)

        if cache_len > 0:
            idx = self._attn_idx_cache.view(1, cache_len)
            valid_cache = idx >= (cache_len - attn_count)
            valid_last = torch.ones((B, 1), device=x_t.device, dtype=torch.bool)
            valid = torch.cat((valid_cache, valid_last), dim=1)
            min_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(~valid.unsqueeze(1).unsqueeze(1), min_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        if cache_len > 0:
            attn_weights_hist = attn_weights[..., :cache_len]
            attn_weights_cur = attn_weights[..., cache_len:]
            out = torch.matmul(attn_weights_hist, attn_v_cache) + attn_weights_cur * v_t
        else:
            out = attn_weights * v_t

        split_sizes = [self.att.ang_v_head_dim, self.att.ang_v_head_dim, self.att.amp_v_head_dim]
        out_ang_r, out_ang_i, out_amp = torch.split(out, split_sizes, dim=-1)

        def merge_head(x):
            return x.transpose(1, 2).contiguous().view(B, 1, -1)

        out_amp = merge_head(out_amp)
        out_ang_r = merge_head(out_ang_r)
        out_ang_i = merge_head(out_ang_i)

        x_amp_att = self.att.to_out_amp(out_amp)
        x_ang_r_att, x_ang_i_att = self.att.to_out_ang(out_ang_r, out_ang_i)

        x_amp = x_amp + x_amp_att
        x_ang_r = x_ang_r + x_ang_r_att
        x_ang_i = x_ang_i + x_ang_i_att

        x_amp_n2 = self.norm2(x_amp)
        x_amp_ffn, amp_gru_cache_out = self.amp_ffn.forward_stream(x_amp_n2, amp_gru_cache)
        x_amp = self.norm3(x_amp + x_amp_ffn)

        x_ang_r_n2, x_ang_i_n2 = self.cnorm2(x_ang_r, x_ang_i)
        (
            x_ang_r_ffn,
            x_ang_i_ffn,
            ang_in_cache_r_out,
            ang_in_cache_i_out,
            ang_mid_cache_r_out,
            ang_mid_cache_i_out,
        ) = self.ang_ffn.forward_stream(
            x_ang_r_n2,
            x_ang_i_n2,
            ang_in_cache_r,
            ang_in_cache_i,
            ang_mid_cache_r,
            ang_mid_cache_i,
        )
        x_ang_r, x_ang_i = self.cnorm3(x_ang_r + x_ang_r_ffn, x_ang_i + x_ang_i_ffn)

        if cache_len > 0:
            evicted_k = attn_k_cache[:, :, :1, :]
            evicted_v = attn_v_cache[:, :, :1, :]
            cur_kv = torch.matmul(k_t.transpose(-2, -1), v_t)
            evicted_kv = torch.matmul(evicted_k.transpose(-2, -1), evicted_v)
            if torch.is_grad_enabled():
                attn_kv_cache_out = attn_kv_cache + cur_kv - evicted_kv
                attn_k_cache_out = torch.roll(attn_k_cache, shifts=-1, dims=2)
                attn_v_cache_out = torch.roll(attn_v_cache, shifts=-1, dims=2)
                attn_k_cache_out[:, :, -1:, :].copy_(k_t)
                attn_v_cache_out[:, :, -1:, :].copy_(v_t)
            else:
                attn_kv_cache_out = attn_kv_cache
                attn_kv_cache_out.add_(cur_kv)
                attn_kv_cache_out.sub_(evicted_kv)
                attn_k_cache_out = torch.roll(attn_k_cache, shifts=-1, dims=2)
                attn_v_cache_out = torch.roll(attn_v_cache, shifts=-1, dims=2)
                attn_k_cache_out[:, :, -1:, :].copy_(k_t)
                attn_v_cache_out[:, :, -1:, :].copy_(v_t)
            attn_count_out = torch.clamp(attn_count + 1.0, max=float(cache_len))
        else:
            attn_k_cache_out = attn_k_cache
            attn_v_cache_out = attn_v_cache
            attn_kv_cache_out = attn_kv_cache
            attn_count_out = attn_count

        out = torch.cat((x_ang_r, x_ang_i, x_amp), dim=-1)
        return (
            out,
            attn_k_cache_out,
            attn_v_cache_out,
            attn_kv_cache_out,
            attn_count_out,
            amp_gru_cache_out,
            ang_in_cache_r_out,
            ang_in_cache_i_out,
            ang_mid_cache_r_out,
            ang_mid_cache_i_out,
        )

def main():
    x = torch.randn(4, 64, 401, 201)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f*t, c)
    transformer = TransformerBlock(d_model=64, n_heads=4)
    x = transformer(x)
    x =  x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())

if __name__ == '__main__':
    main()