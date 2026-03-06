import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerBlock
from pesq import pesq
from joblib import Parallel, delayed
import math


def _shift_append_cache_dim2(cache, new_t):
    if cache.shape[2] == 0:
        return cache
    cache_out = torch.roll(cache, shifts=-1, dims=2)
    cache_out[:, :, -1:, :].copy_(new_t)
    return cache_out


class CwiseRMSNorm(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 n_freqs: int,
                 affine = True,
                 rms_window: int = 100,
                 prior_frames: int = 10,
                 prior_rms: float = 0.1,
                 rms_floor: float = 1e-3,
                 max_gain: float = 6.0,
                 ):
        super(CwiseRMSNorm, self).__init__()
        self.nband = n_freqs
        self.feature_dim = feature_dim
        self.affine = affine
        self.rms_window = rms_window
        self.prior_frames = prior_frames
        self.prior_rms = prior_rms
        self.rms_floor = rms_floor
        self.max_gain = max_gain
        self.eps = 1e-5
        self.register_buffer("_prior_count", torch.tensor(float(prior_frames)), persistent=False)
        self.register_buffer("_prior_power", torch.tensor(float(prior_rms ** 2)), persistent=False)
        self.register_buffer("_min_power", torch.tensor(float(rms_floor ** 2)), persistent=False)
        if affine:
            self.gain_matrix = nn.Parameter(torch.ones([1, feature_dim, 1, 1]))
            self.bias_matrix = nn.Parameter(torch.zeros([1, feature_dim, 1, 1]))
        else:
            self.gain_matrix = None
            self.bias_matrix = None    

    def forward(self, input, return_gain: bool = False):
            # input: (B, C, T, F)
            # channel-independent power per time step, averaged over frequency -> (B, C, T, 1)
            power_t = torch.mean(input ** 2, dim=3, keepdim=True)
            cumulative_sum = torch.cumsum(power_t, dim=2)
            t = input.shape[2]
            if self.rms_window is not None and self.rms_window > 0:
                window = min(self.rms_window, t)
                padded_sum = torch.nn.functional.pad(cumulative_sum, (0, 0, window, 0))
                shifted = padded_sum[:, :, :-window, :]
                running_sum = cumulative_sum - shifted
                divisors = torch.arange(1, t + 1, device=input.device, dtype=input.dtype)
                divisors = torch.clamp(divisors, max=window).view(1, 1, t, 1)
                running_mean_power = running_sum / divisors
            else:
                divisors = torch.arange(1, t + 1, device=input.device, dtype=input.dtype).view(1, 1, t, 1)
                running_mean_power = cumulative_sum / divisors

            if self.prior_frames > 0:
                prior_count = self._prior_count.to(device=input.device, dtype=input.dtype).view(1, 1, 1, 1)
                prior_power = self._prior_power.to(device=input.device, dtype=input.dtype).view(1, 1, 1, 1)
                running_mean_power = (running_mean_power * divisors + prior_power * prior_count) / (divisors + prior_count)

            min_power = self._min_power.to(device=input.device, dtype=input.dtype)
            running_mean_power = torch.clamp(running_mean_power, min=min_power)
            gain = torch.rsqrt(running_mean_power + self.eps)
            gain = torch.clamp(gain, max=self.max_gain)

            input = input * gain
            if self.affine:
                input = input * self.gain_matrix + self.bias_matrix

            if return_gain:
                return input, gain

            return input

    def forward_stream(self, input_t, power_cache, count_cache, return_gain=False):
            # input_t: (B, C, 1, F)
            # power_cache: (B, C, W-1, 1)
            # count_cache: (B, 1, 1, 1), valid history length in [0, W-1]
            power_t = torch.mean(input_t ** 2, dim=3, keepdim=True)
            hist_sum = torch.sum(power_cache, dim=2, keepdim=True)

            divisors = count_cache + 1.0
            running_mean_power = (hist_sum + power_t) / torch.clamp(divisors, min=1.0)

            if self.prior_frames > 0:
                prior_count = self._prior_count.to(device=input_t.device, dtype=input_t.dtype).view(1, 1, 1, 1)
                prior_power = self._prior_power.to(device=input_t.device, dtype=input_t.dtype).view(1, 1, 1, 1)
                running_mean_power = (running_mean_power * divisors + prior_power * prior_count) / (divisors + prior_count)

            min_power = self._min_power.to(device=input_t.device, dtype=input_t.dtype)
            running_mean_power = torch.clamp(running_mean_power, min=min_power)
            gain = torch.rsqrt(running_mean_power + self.eps)
            gain = torch.clamp(gain, max=self.max_gain)

            output_t = input_t * gain
            if self.affine:
                output_t = output_t * self.gain_matrix + self.bias_matrix

            if power_cache.shape[2] > 0:
                power_cache_out = _shift_append_cache_dim2(power_cache, power_t)
                count_cache_out = torch.clamp(count_cache + 1.0, max=float(power_cache.shape[2]))
            else:
                power_cache_out = power_cache
                count_cache_out = count_cache

            if return_gain:
                return output_t, power_cache_out, count_cache_out, gain

            return output_t, power_cache_out, count_cache_out

class CFWiseComplexRMSNorm(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 n_freqs: int,
                 rms_window: int = 100,
                 prior_frames: int = 10,
                 prior_rms: float = 0.1,
                 rms_floor: float = 1e-3,
                 max_gain: float = 6.0,
                 ):
        super(CFWiseComplexRMSNorm, self).__init__()
        self.nband = n_freqs
        self.feature_dim = feature_dim
        self.rms_window = rms_window
        self.prior_frames = prior_frames
        self.prior_rms = prior_rms
        self.rms_floor = rms_floor
        self.max_gain = max_gain
        self.register_buffer("_prior_count", torch.tensor(float(prior_frames)), persistent=False)
        self.register_buffer("_prior_power", torch.tensor(float(prior_rms ** 2)), persistent=False)
        self.register_buffer("_min_power", torch.tensor(float(rms_floor ** 2)), persistent=False)
        self.gain_matrix = nn.Parameter(torch.ones([1, feature_dim, 1, n_freqs]))

    def forward(self, input_r, input_i):
        input_amp = torch.sqrt(input_r ** 2 + input_i ** 2 + 1e-9)
        # input_*: (B, C, T, F)
        # causal running mean power over time using frequency-only statistics per channel
        power_t = torch.mean(input_amp ** 2, dim=3, keepdim=True)
        cumulative_sum = torch.cumsum(power_t, dim=2)
        t = input_amp.shape[2]
        if self.rms_window is not None and self.rms_window > 0:
            window = min(self.rms_window, t)
            padded_sum = torch.nn.functional.pad(cumulative_sum, (0, 0, window, 0))
            shifted = padded_sum[:, :, :-window, :]
            running_sum = cumulative_sum - shifted
            divisors = torch.arange(1, t + 1, device=input_amp.device, dtype=input_amp.dtype)
            divisors = torch.clamp(divisors, max=window).view(1, 1, t, 1)
            running_mean_power = running_sum / divisors
        else:
            divisors = torch.arange(1, t + 1, device=input_amp.device, dtype=input_amp.dtype).view(1, 1, t, 1)
            running_mean_power = cumulative_sum / divisors

        if self.prior_frames > 0:
            prior_count = self._prior_count.to(device=input_amp.device, dtype=input_amp.dtype).view(1, 1, 1, 1)
            prior_power = self._prior_power.to(device=input_amp.device, dtype=input_amp.dtype).view(1, 1, 1, 1)
            running_mean_power = (running_mean_power * divisors + prior_power * prior_count) / (divisors + prior_count)

        min_power = self._min_power.to(device=input_amp.device, dtype=input_amp.dtype)
        running_mean_power = torch.clamp(running_mean_power, min=min_power)
        gain = torch.rsqrt(running_mean_power + 1e-7)
        gain = torch.clamp(gain, max=self.max_gain)

        input_r = input_r * gain
        input_i = input_i * gain
        output_real = input_r * self.gain_matrix
        output_imag = input_i * self.gain_matrix
        return output_real, output_imag

    def forward_stream(self, input_r_t, input_i_t, power_cache, count_cache):
        # input_*_t: (B, C, 1, F)
        # power_cache: (B, C, W-1, 1)
        # count_cache: (B, 1, 1, 1), valid history length in [0, W-1]
        input_amp_t = torch.sqrt(input_r_t ** 2 + input_i_t ** 2 + 1e-9)
        power_t = torch.mean(input_amp_t ** 2, dim=3, keepdim=True)
        hist_sum = torch.sum(power_cache, dim=2, keepdim=True)

        divisors = count_cache + 1.0
        running_mean_power = (hist_sum + power_t) / torch.clamp(divisors, min=1.0)

        if self.prior_frames > 0:
            prior_count = self._prior_count.to(device=input_r_t.device, dtype=input_r_t.dtype).view(1, 1, 1, 1)
            prior_power = self._prior_power.to(device=input_r_t.device, dtype=input_r_t.dtype).view(1, 1, 1, 1)
            running_mean_power = (running_mean_power * divisors + prior_power * prior_count) / (divisors + prior_count)

        min_power = self._min_power.to(device=input_r_t.device, dtype=input_r_t.dtype)
        running_mean_power = torch.clamp(running_mean_power, min=min_power)
        gain = torch.rsqrt(running_mean_power + 1e-7)
        gain = torch.clamp(gain, max=self.max_gain)

        output_real = input_r_t * gain * self.gain_matrix
        output_imag = input_i_t * gain * self.gain_matrix

        if power_cache.shape[2] > 0:
            power_cache_out = _shift_append_cache_dim2(power_cache, power_t)
            count_cache_out = torch.clamp(count_cache + 1.0, max=float(power_cache.shape[2]))
        else:
            power_cache_out = power_cache
            count_cache_out = count_cache

        return output_real, output_imag, power_cache_out, count_cache_out

class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(ComplexConv, self).__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs, bias=False)
        

    def forward(self, real, imag):
        real_conv_real = self.conv_re(real)
        real_conv_imag = self.conv_re(imag)
        imag_conv_real = self.conv_im(real)
        imag_conv_imag = self.conv_im(imag)
        
        real_ = real_conv_real - imag_conv_imag
        imaginary_ = real_conv_imag + imag_conv_real
        
        return real_, imaginary_

class SPConvTranspose2dComplex(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, r=1, **kwargs):
        super(SPConvTranspose2dComplex, self).__init__()
        self.pad1 = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.)
        self.out_channels = out_channels
        self.conv = ComplexConv(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), **kwargs)
        self.r = r

    def forward(self, real, imag):
        real = self.pad1(real)
        imag = self.pad1(imag)
        real_out, imag_out = self.conv(real, imag)
        batch_size, nchannels, H, W = real_out.shape
        real_out = real_out.view((batch_size, self.r, nchannels // self.r, H, W))
        real_out = real_out.permute(0, 2, 3, 4, 1)
        real_out = real_out.contiguous().view((batch_size, nchannels // self.r, H, -1))

        imag_out = imag_out.view((batch_size, self.r, nchannels // self.r, H, W))
        imag_out = imag_out.permute(0, 2, 3, 4, 1)
        imag_out = imag_out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return real_out, imag_out

class LearnableSigmoid3d(nn.Module):
    def __init__(self, in_features_1, in_features_2, initial_beta=3.0):
        super().__init__()
        param_shape_original = (in_features_1, 1, in_features_2)
        self.slope = nn.Parameter(torch.ones(param_shape_original))
        self.slope.requiresGrad = True
        self.beta = initial_beta

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1, **kwargs):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), **kwargs)
        self.r = r
        
    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class InteConvBlock(nn.Module):

    def __init__(self,  kernel_size, amp_in_chn=1, ang_in_chn=1, amp_out_chn=1, ang_out_chn=1, n_freqs=201, separate_grad=False, simple=False, **kwargs):
        super().__init__()
        self.ang_in_chn = ang_in_chn
        self.amp_in_chn = amp_in_chn

        self.amp_out_chn = amp_out_chn
        self.ang_out_chn = ang_out_chn

        self.separate_grad = separate_grad
        self.simple = simple
        self.conv_amp = nn.Conv2d(amp_in_chn, amp_out_chn, kernel_size, **kwargs)
        self.conv_ang = ComplexConv(ang_in_chn, ang_out_chn, kernel_size, **kwargs)
        self.norm_amp = CwiseRMSNorm(amp_out_chn, n_freqs, affine=True)
        self.norm_ang = CFWiseComplexRMSNorm(ang_out_chn, n_freqs)
        self.act_amp = nn.SiLU()

        if not self.simple:
            self.pconv_ang2amp = nn.Conv2d(ang_out_chn, amp_out_chn, kernel_size=(1,1))
            self.pconv_amp2ang = nn.Conv2d(amp_out_chn, ang_out_chn, kernel_size=(1,1))
            torch.nn.init.constant_(self.pconv_ang2amp.weight, 0)
            torch.nn.init.constant_(self.pconv_amp2ang.weight, 0)
            torch.nn.init.constant_(self.pconv_ang2amp.bias, -math.log(2))
            torch.nn.init.constant_(self.pconv_amp2ang.bias, -math.log(2))

            self.act_ang2amp = LearnableSigmoid3d(amp_out_chn, n_freqs, initial_beta=3.0)
            self.act_amp2ang = LearnableSigmoid3d(ang_out_chn, n_freqs, initial_beta=3.0)
        else:
            self.pconv_ang2amp=None
            self.pconv_amp2ang=None
            self.act_ang2amp=None
            self.act_amp2ang=None

    def forward(self, x):
        cos_in, sin_in, std_in = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out, sin_out = self.norm_ang(cos_out, sin_out)
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)
        std_out = self.act_amp(self.norm_amp(self.conv_amp(std_in)))
        if not self.simple:
            if self.separate_grad:
                w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out.detach())))
                w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out.detach())))

            else:
                w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out)))
                w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out)))
            return torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)
        else:
            return torch.cat((cos_out, sin_out, std_out), dim=1)

    def forward_stream(self, x_ctx, amp_power_cache, amp_count_cache, ang_power_cache, ang_count_cache):
        cos_in, sin_in, std_in = torch.split(x_ctx, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out = cos_out[:, :, -1:, :]
        sin_out = sin_out[:, :, -1:, :]
        cos_out, sin_out, ang_power_cache_out, ang_count_cache_out = self.norm_ang.forward_stream(
            cos_out, sin_out, ang_power_cache, ang_count_cache
        )
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)

        std_out = self.conv_amp(std_in)
        std_out = std_out[:, :, -1:, :]
        std_out, amp_power_cache_out, amp_count_cache_out = self.norm_amp.forward_stream(
            std_out, amp_power_cache, amp_count_cache
        )
        std_out = self.act_amp(std_out)

        if not self.simple:
            if self.separate_grad:
                w_amp2ang = self.act_amp2ang(self.pconv_amp2ang(std_out.detach()))
                w_ang2amp = self.act_ang2amp(self.pconv_ang2amp(ang_amp_out.detach()))
            else:
                w_amp2ang = self.act_amp2ang(self.pconv_amp2ang(std_out))
                w_ang2amp = self.act_ang2amp(self.pconv_ang2amp(ang_amp_out))
            out = torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)
        else:
            out = torch.cat((cos_out, sin_out, std_out), dim=1)

        return out, amp_power_cache_out, amp_count_cache_out, ang_power_cache_out, ang_count_cache_out

class InteConvBlockTranspose(nn.Module):
    def __init__(self, kernel_size, amp_in_chn=1, ang_in_chn=1, amp_out_chn=1, ang_out_chn=1, n_freqs=129, separate_grad=False, **kwargs):
        super().__init__()

        self.ang_in_chn = ang_in_chn
        self.amp_in_chn = amp_in_chn
        self.amp_out_chn = amp_out_chn
        self.ang_out_chn = ang_out_chn

        self.conv_amp = SPConvTranspose2d(amp_in_chn, amp_out_chn, kernel_size, **kwargs)
        self.conv_ang = SPConvTranspose2dComplex(ang_in_chn, ang_out_chn, kernel_size, **kwargs)

        self.pconv_ang2amp = nn.Conv2d(ang_out_chn, amp_out_chn, 1)
        self.pconv_amp2ang = nn.Conv2d(amp_out_chn, ang_out_chn, 1)
        torch.nn.init.constant_(self.pconv_ang2amp.weight, 0)
        torch.nn.init.constant_(self.pconv_amp2ang.weight, 0)
        torch.nn.init.constant_(self.pconv_ang2amp.bias, -math.log(2))
        torch.nn.init.constant_(self.pconv_amp2ang.bias, -math.log(2))
        self.norm_amp = CwiseRMSNorm(amp_out_chn, n_freqs, affine=True)
        self.norm_ang = CFWiseComplexRMSNorm(ang_out_chn, n_freqs)

        self.act_amp = nn.SiLU()
        self.act_ang2amp = LearnableSigmoid3d(amp_out_chn, n_freqs, initial_beta=3.0)
        self.act_amp2ang = LearnableSigmoid3d(ang_out_chn, n_freqs, initial_beta=3.0)
        self.separate_grad = separate_grad
    def forward(self, x):
        cos_in, sin_in, std_in = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out, sin_out = self.norm_ang(cos_out, sin_out)
        std_out = self.act_amp(self.norm_amp(self.conv_amp(std_in)))
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)

        if self.separate_grad:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out.detach())))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out.detach())))
            # print(w_ang2amp)
        else:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out)))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out)))

        return torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)

    def forward_stream(self, x_t, amp_power_cache, amp_count_cache, ang_power_cache, ang_count_cache):
        cos_in, sin_in, std_in = torch.split(x_t, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)

        cos_out, sin_out = self.conv_ang(cos_in, sin_in)
        cos_out, sin_out, ang_power_cache_out, ang_count_cache_out = self.norm_ang.forward_stream(
            cos_out, sin_out, ang_power_cache, ang_count_cache
        )

        std_out = self.conv_amp(std_in)
        std_out, amp_power_cache_out, amp_count_cache_out = self.norm_amp.forward_stream(
            std_out, amp_power_cache, amp_count_cache
        )
        std_out = self.act_amp(std_out)
        ang_amp_out = torch.sqrt(cos_out ** 2 + sin_out ** 2 + 1e-9)

        if self.separate_grad:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out.detach())))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out.detach())))
        else:
            w_amp2ang = self.act_amp2ang((self.pconv_amp2ang(std_out)))
            w_ang2amp = self.act_ang2amp((self.pconv_ang2amp(ang_amp_out)))

        out = torch.cat((w_amp2ang * cos_out, w_amp2ang * sin_out, w_ang2amp * std_out), dim=1)
        return out, amp_power_cache_out, amp_count_cache_out, ang_power_cache_out, ang_count_cache_out

class TSTransformerBlock(nn.Module):
    def __init__(self,h):
        super(TSTransformerBlock, self).__init__()
        attn_lookback = getattr(h, 'time_attn_lookback', 100)
        self.time_transformer = TransformerBlock(h, causal=True, attn_lookback=attn_lookback)
        self.freq_transformer = TransformerBlock(h, causal=False)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        return x

class DenseBlock(nn.Module):
    def __init__(self, kernel_size=(2, 3), depth=4, amp_in_chn=48, ang_in_chn=16, n_freqs=201, separate_grad=False):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.amp_in_chn = amp_in_chn
        self.ang_in_chn = ang_in_chn
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
                        
            dense_conv_fan = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                InteConvBlock(kernel_size, 
                    amp_in_chn=self.amp_in_chn*(i+1), 
                    ang_in_chn=self.ang_in_chn*(i+1), 
                    amp_out_chn=self.amp_in_chn, 
                    ang_out_chn=self.ang_in_chn,
                    dilation=(dilation, 1), 
                    n_freqs=n_freqs,
                    separate_grad=separate_grad)
            )
            self.dense_block.append(dense_conv_fan)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            x_cos, x_sin, x_std = torch.split(x, [self.ang_in_chn, self.ang_in_chn, self.amp_in_chn], dim=1)
            _, C, _, _ = skip.shape
            skip_cos, skip_sin, skip_std = torch.split(skip, [self.ang_in_chn*(i+1), self.ang_in_chn*(i+1), self.amp_in_chn*(i+1)], dim=1)
            skip = torch.cat([x_cos, skip_cos, x_sin, skip_sin, x_std, skip_std], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, ang_chn=16, amp_chn=48):
        super(DenseEncoder, self).__init__()

        self.dense_conv_1 = InteConvBlock((1,1), amp_in_chn=1, ang_in_chn=1, amp_out_chn=amp_chn, ang_out_chn=ang_chn, n_freqs=201, separate_grad=True, simple=True)
        self.dense_block = DenseBlock(depth=4, amp_in_chn=amp_chn, ang_in_chn=ang_chn, n_freqs=201)
        
        self.dense_conv_3 = InteConvBlock(
            kernel_size=(1, 3), 
            amp_in_chn=amp_chn,
            ang_in_chn=ang_chn,
            amp_out_chn=amp_chn,
            ang_out_chn=ang_chn,
            stride=(1, 2), 
            padding=(0, 1),
            n_freqs=101
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channel=1, ang_chn=16, amp_chn=48):
        super(Decoder, self).__init__()
        self.ang_chn = ang_chn
        self.amp_chn = amp_chn
 
        self.dense_block = DenseBlock(depth=4, ang_in_chn=ang_chn, amp_in_chn=amp_chn, n_freqs=101, separate_grad=False)

        self.upsample_layer = InteConvBlockTranspose(
            kernel_size=(1, 3), 
            amp_in_chn=amp_chn, 
            ang_in_chn=ang_chn,
            amp_out_chn=amp_chn, 
            ang_out_chn=ang_chn,
            r=2,
            n_freqs=202,
            separate_grad=False
        )

        self.amp_conv_layer = nn.Sequential(nn.Conv2d(
            in_channels=self.amp_chn, 
            out_channels=out_channel, 
            kernel_size=(1, 2)),
            nn.ReLU())
  
        self.ang_conv = ComplexConv(self.ang_chn, out_channel, (1,2))

    def forward(self, x):

        x = self.dense_block(x)
        # x = self.pad1d(x)
        x = self.upsample_layer(x)
        x_ang_r, x_ang_i, x_amp = torch.split(x, [self.ang_chn, self.ang_chn, self.amp_chn], dim=1)
        x_r, x_i = self.ang_conv(x_ang_r, x_ang_i)
        x_ang = torch.atan2(x_i + 1e-9, x_r + 1e-9)
        x_ang = x_ang.permute(0, 3, 2, 1).squeeze(-1)

        x_amp = self.amp_conv_layer(x_amp)
        x_amp = x_amp.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x_amp, x_ang
    
class MPNet(nn.Module):

    def __init__(self, h, **kwargs):
        super(MPNet, self).__init__()
        self.num_tscblocks = h.num_tsconformers
        amp_chn = h.amp_chn
        ang_chn = h.ang_chn
        self.dense_encoder = DenseEncoder(in_channel=3, amp_chn=amp_chn, ang_chn=ang_chn)
        self.TSTransformer = nn.ModuleList([TSTransformerBlock(h) for _ in range(self.num_tscblocks)])
        self.decoder = Decoder(out_channel=1, amp_chn=amp_chn, ang_chn=ang_chn)
        # Per-utterance / per-batch magnitude RMS normalization (AGC):
        n_freqs = getattr(h, 'n_freqs', 201)
        self.input_rms = CwiseRMSNorm(feature_dim=1, n_freqs=n_freqs, affine=False)
        
    def forward(self, noisy_amp, noisy_pha):
        # Apply input magnitude RMS normalization (AGC): normalize noisy_amp and record gain
        # noisy_amp: [B, F, T]
        noisy_amp_in = noisy_amp.unsqueeze(1).permute(0, 1, 3, 2)  # [B,1,T,F]
        norm_amp_in, amp_gain = self.input_rms(noisy_amp_in, return_gain=True)
        norm_amp = norm_amp_in.permute(0, 1, 3, 2).squeeze(1)  # [B, F, T]

        # Encoder: use normalized amplitude
        x = torch.stack((torch.cos(noisy_pha), torch.sin(noisy_pha), norm_amp), dim=1) # [B, 3, F, T]
        x = x.permute(0, 1, 3, 2) # [B, C, T, F]

        x_encoded = self.dense_encoder(x)

        # Transformer Blocks
        x_transformed = x_encoded
        for i in range(self.num_tscblocks):
            x_transformed = self.TSTransformer[i](x_transformed)
        # Decoders
        denoised_amp, denoised_pha = self.decoder(x_transformed)
        
        # Reconstruct complex spectrum
        denoised_com = torch.stack(
            (
                denoised_amp * torch.cos(denoised_pha),
                denoised_amp * torch.sin(denoised_pha),
            ),
            dim=-1,
        )

        # Restore original magnitude scale using recorded gain (inverse of normalization)
        # amp_gain: [B,1,T,1] -> squeeze last dim -> [B,1,T], broadcasts over frequency
        denoised_amp = denoised_amp / amp_gain.squeeze(-1)
        denoised_com = denoised_com / amp_gain

        return denoised_amp, denoised_pha, denoised_com


class StreamMPNet(nn.Module):
    """
    Streaming MPNet with module-specific caches:
    - DenseBlock caches: per-layer temporal caches determined by dilation (1/2/4/8)
    - Time-transformer cache: attention history determined by attn_lookback

    The model consumes one STFT frame each call: (B, F, 1).
    """
    def __init__(self, h):
        super(StreamMPNet, self).__init__()
        self.mpnet = MPNet(h)
        self.safe_ts_cache_read = bool(getattr(h, 'safe_ts_cache_read', False))

    def fuse_inference_kernels(self):
        for ts_block in self.mpnet.TSTransformer:
            ts_block.time_transformer.fuse_inference_kernels()
            ts_block.freq_transformer.fuse_inference_kernels()

    def _dense_layer_meta(self, dense_block: DenseBlock):
        base_channels = dense_block.ang_in_chn * 2 + dense_block.amp_in_chn
        layer_channels = [base_channels * (i + 1) for i in range(dense_block.depth)]
        layer_dilations = [2 ** i for i in range(dense_block.depth)]
        return base_channels, layer_channels, layer_dilations

    def init_stream_cache(self, batch_size, n_freqs, device=None, dtype=torch.float32):
        enc_dense = self.mpnet.dense_encoder.dense_block
        _, enc_channels, enc_dilations = self._dense_layer_meta(enc_dense)

        dec_dense = self.mpnet.decoder.dense_block
        _, dec_channels, dec_dilations = self._dense_layer_meta(dec_dense)

        n_freqs_mid = (n_freqs + 1) // 2

        enc_caches = [
            torch.zeros(batch_size, enc_channels[i], enc_dilations[i], n_freqs, device=device, dtype=dtype)
            for i in range(enc_dense.depth)
        ]
        dec_caches = [
            torch.zeros(batch_size, dec_channels[i], dec_dilations[i], n_freqs_mid, device=device, dtype=dtype)
            for i in range(dec_dense.depth)
        ]

        num_ts = len(self.mpnet.TSTransformer)
        attn_lookback = self.mpnet.TSTransformer[0].time_transformer.att.attn_lookback if num_ts > 0 else 1
        ts_cache_len = max(attn_lookback - 1, 0)
        if num_ts > 0:
            att_module = self.mpnet.TSTransformer[0].time_transformer.att
            ts_heads = att_module.num_heads
            ts_k_dim = att_module.ang_qk_head_dim * 2 + att_module.amp_qk_head_dim
            ts_v_dim = att_module.ang_v_head_dim * 2 + att_module.amp_v_head_dim
        else:
            ts_heads = 1
            ts_k_dim = 1
            ts_v_dim = 1

        ts_k_cache = torch.zeros(num_ts, batch_size, n_freqs_mid, ts_heads, ts_cache_len, ts_k_dim, device=device, dtype=dtype)
        ts_v_cache = torch.zeros(num_ts, batch_size, n_freqs_mid, ts_heads, ts_cache_len, ts_v_dim, device=device, dtype=dtype)
        ts_kv_cache = torch.zeros(num_ts, batch_size, n_freqs_mid, ts_heads, ts_k_dim, ts_v_dim, device=device, dtype=dtype)
        ts_attn_count = torch.zeros(num_ts, batch_size, n_freqs_mid, 1, device=device, dtype=dtype)

        if num_ts > 0:
            amp_hidden = self.mpnet.TSTransformer[0].time_transformer.amp_ffn.gru.hidden_size
            ang_chn = self.mpnet.TSTransformer[0].time_transformer.ang_chn
            ang_inner = self.mpnet.TSTransformer[0].time_transformer.ang_ffn.chn_inner
            ang_cache_len = self.mpnet.TSTransformer[0].time_transformer.ang_ffn.conv1d_kernel
        else:
            amp_hidden = self.mpnet.decoder.amp_chn * 2
            ang_chn = self.mpnet.decoder.ang_chn
            ang_inner = 64
            ang_cache_len = 4

        ts_gru_cache = torch.zeros(num_ts, 2, batch_size, n_freqs_mid, amp_hidden, device=device, dtype=dtype)
        ts_ang_in_cache_r = torch.zeros(num_ts, batch_size, n_freqs_mid, ang_chn, ang_cache_len, device=device, dtype=dtype)
        ts_ang_in_cache_i = torch.zeros(num_ts, batch_size, n_freqs_mid, ang_chn, ang_cache_len, device=device, dtype=dtype)
        ts_ang_mid_cache_r = torch.zeros(num_ts, batch_size, n_freqs_mid, ang_inner, ang_cache_len, device=device, dtype=dtype)
        ts_ang_mid_cache_i = torch.zeros(num_ts, batch_size, n_freqs_mid, ang_inner, ang_cache_len, device=device, dtype=dtype)

        rms_window = max(int(self.mpnet.dense_encoder.dense_conv_1.norm_amp.rms_window) - 1, 0)
        n_enc_norm = 6
        n_dec_norm = 5
        enc_amp_power_cache = torch.zeros(n_enc_norm, batch_size, self.mpnet.dense_encoder.dense_conv_1.amp_out_chn, rms_window, 1, device=device, dtype=dtype)
        enc_ang_power_cache = torch.zeros(n_enc_norm, batch_size, self.mpnet.dense_encoder.dense_conv_1.ang_out_chn, rms_window, 1, device=device, dtype=dtype)
        enc_amp_count_cache = torch.zeros(n_enc_norm, batch_size, 1, 1, 1, device=device, dtype=dtype)
        enc_ang_count_cache = torch.zeros(n_enc_norm, batch_size, 1, 1, 1, device=device, dtype=dtype)

        dec_amp_power_cache = torch.zeros(n_dec_norm, batch_size, self.mpnet.decoder.dense_block.amp_in_chn, rms_window, 1, device=device, dtype=dtype)
        dec_ang_power_cache = torch.zeros(n_dec_norm, batch_size, self.mpnet.decoder.dense_block.ang_in_chn, rms_window, 1, device=device, dtype=dtype)
        dec_amp_count_cache = torch.zeros(n_dec_norm, batch_size, 1, 1, 1, device=device, dtype=dtype)
        dec_ang_count_cache = torch.zeros(n_dec_norm, batch_size, 1, 1, 1, device=device, dtype=dtype)

        # input AGC caches (power history, count)
        input_rms_window = max(int(self.mpnet.input_rms.rms_window) - 1, 0)
        input_amp_power_cache = torch.zeros(batch_size, 1, input_rms_window, 1, device=device, dtype=dtype)
        input_amp_count_cache = torch.zeros(batch_size, 1, 1, 1, device=device, dtype=dtype)

        return (
            enc_caches[0], enc_caches[1], enc_caches[2], enc_caches[3],
            dec_caches[0], dec_caches[1], dec_caches[2], dec_caches[3],
            ts_k_cache,
            ts_v_cache,
            ts_kv_cache,
            ts_attn_count,
            ts_gru_cache,
            ts_ang_in_cache_r,
            ts_ang_in_cache_i,
            ts_ang_mid_cache_r,
            ts_ang_mid_cache_i,
            enc_amp_power_cache,
            enc_ang_power_cache,
            enc_amp_count_cache,
            enc_ang_count_cache,
            dec_amp_power_cache,
            dec_ang_power_cache,
            dec_amp_count_cache,
            dec_ang_count_cache,
            input_amp_power_cache,
            input_amp_count_cache,
        )

    def _stream_dense_block(self, dense_block: DenseBlock, x_t, caches, amp_power_cache, amp_count_cache, ang_power_cache, ang_count_cache):
        _, _, layer_dilations = self._dense_layer_meta(dense_block)
        skip_t = x_t
        new_caches = []
        new_amp_power = amp_power_cache
        new_amp_count = amp_count_cache
        new_ang_power = ang_power_cache
        new_ang_count = ang_count_cache

        for i in range(dense_block.depth):
            cache_i = caches[i]
            inp_raw = torch.cat([cache_i, skip_t], dim=2)
            inp_i = torch.nn.functional.pad(inp_raw, (1, 1, 0, 0))
            block_i = dense_block.dense_block[i][1]
            out_t, amp_power_i_out, amp_count_i_out, ang_power_i_out, ang_count_i_out = block_i.forward_stream(
                inp_i,
                amp_power_cache[i],
                amp_count_cache[i],
                ang_power_cache[i],
                ang_count_cache[i],
            )

            x_cos, x_sin, x_std = torch.split(out_t, [dense_block.ang_in_chn, dense_block.ang_in_chn, dense_block.amp_in_chn], dim=1)
            skip_cos, skip_sin, skip_std = torch.split(
                skip_t,
                [dense_block.ang_in_chn * (i + 1), dense_block.ang_in_chn * (i + 1), dense_block.amp_in_chn * (i + 1)],
                dim=1,
            )
            skip_t = torch.cat([x_cos, skip_cos, x_sin, skip_sin, x_std, skip_std], dim=1)

            dilation_i = layer_dilations[i]
            new_cache_i = inp_raw[:, :, -dilation_i:, :]
            new_caches.append(new_cache_i)
            new_amp_power[i].copy_(amp_power_i_out)
            new_amp_count[i].copy_(amp_count_i_out)
            new_ang_power[i].copy_(ang_power_i_out)
            new_ang_count[i].copy_(ang_count_i_out)

        return out_t, new_caches, new_amp_power, new_amp_count, new_ang_power, new_ang_count

    def forward(
        self,
        noisy_amp_t,
        noisy_pha_t,
        enc_cache_1,
        enc_cache_2,
        enc_cache_3,
        enc_cache_4,
        dec_cache_1,
        dec_cache_2,
        dec_cache_3,
        dec_cache_4,
        ts_k_cache,
        ts_v_cache,
        ts_kv_cache,
        ts_attn_count,
        ts_gru_cache,
        ts_ang_in_cache_r,
        ts_ang_in_cache_i,
        ts_ang_mid_cache_r,
        ts_ang_mid_cache_i,
        enc_amp_power_cache,
        enc_ang_power_cache,
        enc_amp_count_cache,
        enc_ang_count_cache,
        dec_amp_power_cache,
        dec_ang_power_cache,
        dec_amp_count_cache,
        dec_ang_count_cache,
        input_amp_power_cache,
        input_amp_count_cache,
    ):
        """
        noisy_amp_t/noisy_pha_t: (B, F, 1)
        enc_cache_i: per-layer encoder dense block cache
        dec_cache_i: per-layer decoder dense block cache
        ts_k_cache: (Nts, B, Fmid, H, Lattn, Dk)
        ts_v_cache: (Nts, B, Fmid, H, Lattn, Dv)
        ts_kv_cache: (Nts, B, Fmid, H, Dk, Dv)
        ts_attn_count: (Nts, B, Fmid, 1)
        ts_gru_cache: (Nts, 2, B, Fmid, H)
        ts_ang_*_cache: complex ffn caches for causal conv1d (kernel=4 -> cache=3)
        """
        # Apply streaming AGC to amplitude channel (normalize and record gain)
        # noisy_amp_t: (B, F, 1) -> prepare (B,1,1,F) for input_rms.forward_stream
        noisy_amp_in = noisy_amp_t.permute(0, 2, 1).unsqueeze(2)  # (B,1,1,F)
        amp_out_t, input_amp_power_cache_out, input_amp_count_cache_out, amp_gain = self.mpnet.input_rms.forward_stream(
            noisy_amp_in, input_amp_power_cache, input_amp_count_cache, return_gain=True
        )
        # amp_out_t: (B,1,1,F) -> make (B,F,1) to match previous representation
        amp_norm_t = amp_out_t.squeeze(1).permute(0, 2, 1)  # (B,F,1)

        x = torch.stack((torch.cos(noisy_pha_t), torch.sin(noisy_pha_t), amp_norm_t), dim=1)
        x = x.permute(0, 1, 3, 2)

        x, enc_amp_power_0, enc_amp_count_0, enc_ang_power_0, enc_ang_count_0 = self.mpnet.dense_encoder.dense_conv_1.forward_stream(
            x,
            enc_amp_power_cache[0],
            enc_amp_count_cache[0],
            enc_ang_power_cache[0],
            enc_ang_count_cache[0],
        )
        x, new_enc_caches, enc_amp_power_blk, enc_amp_count_blk, enc_ang_power_blk, enc_ang_count_blk = self._stream_dense_block(
            self.mpnet.dense_encoder.dense_block,
            x,
            [enc_cache_1, enc_cache_2, enc_cache_3, enc_cache_4],
            enc_amp_power_cache[1:5],
            enc_amp_count_cache[1:5],
            enc_ang_power_cache[1:5],
            enc_ang_count_cache[1:5],
        )
        x, enc_amp_power_5, enc_amp_count_5, enc_ang_power_5, enc_ang_count_5 = self.mpnet.dense_encoder.dense_conv_3.forward_stream(
            x,
            enc_amp_power_cache[5],
            enc_amp_count_cache[5],
            enc_ang_power_cache[5],
            enc_ang_count_cache[5],
        )

        new_enc_amp_power_cache = enc_amp_power_cache
        new_enc_amp_count_cache = enc_amp_count_cache
        new_enc_ang_power_cache = enc_ang_power_cache
        new_enc_ang_count_cache = enc_ang_count_cache
        new_enc_amp_power_cache[0].copy_(enc_amp_power_0)
        new_enc_amp_power_cache[1:5].copy_(enc_amp_power_blk)
        new_enc_amp_power_cache[5].copy_(enc_amp_power_5)
        new_enc_amp_count_cache[0].copy_(enc_amp_count_0)
        new_enc_amp_count_cache[1:5].copy_(enc_amp_count_blk)
        new_enc_amp_count_cache[5].copy_(enc_amp_count_5)
        new_enc_ang_power_cache[0].copy_(enc_ang_power_0)
        new_enc_ang_power_cache[1:5].copy_(enc_ang_power_blk)
        new_enc_ang_power_cache[5].copy_(enc_ang_power_5)
        new_enc_ang_count_cache[0].copy_(enc_ang_count_0)
        new_enc_ang_count_cache[1:5].copy_(enc_ang_count_blk)
        new_enc_ang_count_cache[5].copy_(enc_ang_count_5)

        b, c, t, f = x.size()
        new_ts_k_cache = ts_k_cache
        new_ts_v_cache = ts_v_cache
        new_ts_kv_cache = ts_kv_cache
        new_ts_attn_count = ts_attn_count
        new_ts_gru_cache = ts_gru_cache
        new_ts_ang_in_cache_r = ts_ang_in_cache_r
        new_ts_ang_in_cache_i = ts_ang_in_cache_i
        new_ts_ang_mid_cache_r = ts_ang_mid_cache_r
        new_ts_ang_mid_cache_i = ts_ang_mid_cache_i
        for i, ts_block in enumerate(self.mpnet.TSTransformer):
            x_btfc = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)

            if self.safe_ts_cache_read:
                attn_k_cache_src = ts_k_cache[i].clone()
                attn_v_cache_src = ts_v_cache[i].clone()
                attn_kv_cache_src = ts_kv_cache[i].clone()
                attn_count_src = ts_attn_count[i].clone()
                gru_cache_src = ts_gru_cache[i].clone()
                ang_in_cache_r_src = ts_ang_in_cache_r[i].clone()
                ang_in_cache_i_src = ts_ang_in_cache_i[i].clone()
                ang_mid_cache_r_src = ts_ang_mid_cache_r[i].clone()
                ang_mid_cache_i_src = ts_ang_mid_cache_i[i].clone()
            else:
                attn_k_cache_src = ts_k_cache[i]
                attn_v_cache_src = ts_v_cache[i]
                attn_kv_cache_src = ts_kv_cache[i]
                attn_count_src = ts_attn_count[i]
                gru_cache_src = ts_gru_cache[i]
                ang_in_cache_r_src = ts_ang_in_cache_r[i]
                ang_in_cache_i_src = ts_ang_in_cache_i[i]
                ang_mid_cache_r_src = ts_ang_mid_cache_r[i]
                ang_mid_cache_i_src = ts_ang_mid_cache_i[i]

            attn_k_cache_i = attn_k_cache_src.contiguous().view(b * f, ts_k_cache.shape[3], ts_k_cache.shape[4], ts_k_cache.shape[5])
            attn_v_cache_i = attn_v_cache_src.contiguous().view(b * f, ts_v_cache.shape[3], ts_v_cache.shape[4], ts_v_cache.shape[5])
            attn_kv_cache_i = attn_kv_cache_src.contiguous().view(b * f, ts_kv_cache.shape[3], ts_kv_cache.shape[4], ts_kv_cache.shape[5])
            attn_count_i = attn_count_src.contiguous().view(b * f, 1)
            gru_cache_i = gru_cache_src.contiguous().view(2, b * f, ts_gru_cache.shape[-1])
            ang_in_cache_r_i = ang_in_cache_r_src.contiguous().view(b * f, ts_ang_in_cache_r.shape[3], ts_ang_in_cache_r.shape[4])
            ang_in_cache_i_i = ang_in_cache_i_src.contiguous().view(b * f, ts_ang_in_cache_i.shape[3], ts_ang_in_cache_i.shape[4])
            ang_mid_cache_r_i = ang_mid_cache_r_src.contiguous().view(b * f, ts_ang_mid_cache_r.shape[3], ts_ang_mid_cache_r.shape[4])
            ang_mid_cache_i_i = ang_mid_cache_i_src.contiguous().view(b * f, ts_ang_mid_cache_i.shape[3], ts_ang_mid_cache_i.shape[4])

            (
                y_t,
                attn_k_cache_i_out,
                attn_v_cache_i_out,
                attn_kv_cache_i_out,
                attn_count_i_out,
                gru_cache_i_out,
                ang_in_cache_r_i_out,
                ang_in_cache_i_i_out,
                ang_mid_cache_r_i_out,
                ang_mid_cache_i_i_out,
            ) = ts_block.time_transformer.forward_stream(
                x_btfc,
                attn_k_cache_i,
                attn_v_cache_i,
                attn_kv_cache_i,
                attn_count_i,
                gru_cache_i,
                ang_in_cache_r_i,
                ang_in_cache_i_i,
                ang_mid_cache_r_i,
                ang_mid_cache_i_i,
            )
            y_t = y_t + x_btfc

            new_ts_k_cache[i].copy_(attn_k_cache_i_out.view(b, f, ts_k_cache.shape[3], ts_k_cache.shape[4], ts_k_cache.shape[5]))
            new_ts_v_cache[i].copy_(attn_v_cache_i_out.view(b, f, ts_v_cache.shape[3], ts_v_cache.shape[4], ts_v_cache.shape[5]))
            new_ts_kv_cache[i].copy_(attn_kv_cache_i_out.view(b, f, ts_kv_cache.shape[3], ts_kv_cache.shape[4], ts_kv_cache.shape[5]))
            new_ts_attn_count[i].copy_(attn_count_i_out.view(b, f, 1))
            new_ts_gru_cache[i].copy_(gru_cache_i_out.view(2, b, f, ts_gru_cache.shape[-1]))
            new_ts_ang_in_cache_r[i].copy_(ang_in_cache_r_i_out.view(b, f, ts_ang_in_cache_r.shape[3], ts_ang_in_cache_r.shape[4]))
            new_ts_ang_in_cache_i[i].copy_(ang_in_cache_i_i_out.view(b, f, ts_ang_in_cache_i.shape[3], ts_ang_in_cache_i.shape[4]))
            new_ts_ang_mid_cache_r[i].copy_(ang_mid_cache_r_i_out.view(b, f, ts_ang_mid_cache_r.shape[3], ts_ang_mid_cache_r.shape[4]))
            new_ts_ang_mid_cache_i[i].copy_(ang_mid_cache_i_i_out.view(b, f, ts_ang_mid_cache_i.shape[3], ts_ang_mid_cache_i.shape[4]))

            y = y_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
            y = ts_block.freq_transformer(y) + y
            x = y.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()

        if len(self.mpnet.TSTransformer) == 0:
            new_ts_k_cache = ts_k_cache
            new_ts_v_cache = ts_v_cache
            new_ts_kv_cache = ts_kv_cache
            new_ts_attn_count = ts_attn_count
            new_ts_gru_cache = ts_gru_cache
            new_ts_ang_in_cache_r = ts_ang_in_cache_r
            new_ts_ang_in_cache_i = ts_ang_in_cache_i
            new_ts_ang_mid_cache_r = ts_ang_mid_cache_r
            new_ts_ang_mid_cache_i = ts_ang_mid_cache_i

        x, new_dec_caches, dec_amp_power_blk, dec_amp_count_blk, dec_ang_power_blk, dec_ang_count_blk = self._stream_dense_block(
            self.mpnet.decoder.dense_block,
            x,
            [dec_cache_1, dec_cache_2, dec_cache_3, dec_cache_4],
            dec_amp_power_cache[0:4],
            dec_amp_count_cache[0:4],
            dec_ang_power_cache[0:4],
            dec_ang_count_cache[0:4],
        )

        x, dec_amp_power_4, dec_amp_count_4, dec_ang_power_4, dec_ang_count_4 = self.mpnet.decoder.upsample_layer.forward_stream(
            x,
            dec_amp_power_cache[4],
            dec_amp_count_cache[4],
            dec_ang_power_cache[4],
            dec_ang_count_cache[4],
        )

        new_dec_amp_power_cache = dec_amp_power_cache
        new_dec_amp_count_cache = dec_amp_count_cache
        new_dec_ang_power_cache = dec_ang_power_cache
        new_dec_ang_count_cache = dec_ang_count_cache
        new_dec_amp_power_cache[0:4].copy_(dec_amp_power_blk)
        new_dec_amp_power_cache[4].copy_(dec_amp_power_4)
        new_dec_amp_count_cache[0:4].copy_(dec_amp_count_blk)
        new_dec_amp_count_cache[4].copy_(dec_amp_count_4)
        new_dec_ang_power_cache[0:4].copy_(dec_ang_power_blk)
        new_dec_ang_power_cache[4].copy_(dec_ang_power_4)
        new_dec_ang_count_cache[0:4].copy_(dec_ang_count_blk)
        new_dec_ang_count_cache[4].copy_(dec_ang_count_4)
        x_ang_r, x_ang_i, x_amp = torch.split(x, [self.mpnet.decoder.ang_chn, self.mpnet.decoder.ang_chn, self.mpnet.decoder.amp_chn], dim=1)
        x_r, x_i = self.mpnet.decoder.ang_conv(x_ang_r, x_ang_i)
        denoised_pha_t = torch.atan2(x_i + 1e-9, x_r + 1e-9).permute(0, 3, 2, 1).squeeze(-1)

        denoised_amp_t = self.mpnet.decoder.amp_conv_layer(x_amp).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com_t = torch.stack(
            (
                denoised_amp_t * torch.cos(denoised_pha_t),
                denoised_amp_t * torch.sin(denoised_pha_t),
            ),
            dim=-1,
        )

        # Restore original magnitude scale using recorded streaming gain
        denoised_amp_t = denoised_amp_t / amp_gain.squeeze(-1)
        denoised_com_t = denoised_com_t / amp_gain

        return (
            denoised_amp_t,
            denoised_pha_t,
            denoised_com_t,
            new_enc_caches[0], new_enc_caches[1], new_enc_caches[2], new_enc_caches[3],
            new_dec_caches[0], new_dec_caches[1], new_dec_caches[2], new_dec_caches[3],
            new_ts_k_cache,
            new_ts_v_cache,
            new_ts_kv_cache,
            new_ts_attn_count,
            new_ts_gru_cache,
            new_ts_ang_in_cache_r,
            new_ts_ang_in_cache_i,
            new_ts_ang_mid_cache_r,
            new_ts_ang_mid_cache_i,
            new_enc_amp_power_cache,
            new_enc_ang_power_cache,
            new_enc_amp_count_cache,
            new_enc_ang_count_cache,
            new_dec_amp_power_cache,
            new_dec_ang_power_cache,
            new_dec_amp_count_cache,
            new_dec_ang_count_cache,
            input_amp_power_cache_out,
            input_amp_count_cache_out,
        )
    
def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss, gd_loss, iaf_loss
    
def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            16000)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score

def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1

    return pesq_score
