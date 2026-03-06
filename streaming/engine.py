import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
from models.model import StreamMPNet


class MPNetStreamingEnhancer:
    def __init__(
        self,
        checkpoint_file: str,
        config_file: Optional[str] = None,
        device: str = "cpu",
        num_threads: Optional[int] = None,
        use_cuda_graph: bool = False,
    ):
        self.checkpoint_file = Path(checkpoint_file)
        if config_file is None:
            config_file = str(self.checkpoint_file.parent / "config.json")
        self.config_file = Path(config_file)

        with open(self.config_file, "r", encoding="utf-8") as f:
            self.h = AttrDict(json.loads(f.read()))

        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        if self.device.type == "cpu":
            if num_threads is None:
                num_threads = max(1, (os.cpu_count() or 4) // 2)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(1)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        state = torch.load(self.checkpoint_file, map_location=self.device)
        self.model = StreamMPNet(self.h).to(self.device).eval()

        ckpt_state = state["generator"]
        model_state = self.model.mpnet.state_dict()

        matched_state = {}
        skipped_missing_or_mismatch = 0
        for key, value in ckpt_state.items():
            model_value = model_state.get(key)
            if model_value is None:
                skipped_missing_or_mismatch += 1
                continue
            if model_value.shape != value.shape:
                if 'TSTransformer' in key and 'deconv1d' in key and 'ang_ffn' in key:
                    value = value.unsqueeze(-2)
                else:
                    skipped_missing_or_mismatch += 1
                    continue
            matched_state[key] = value

        load_result = self.model.mpnet.load_state_dict(matched_state, strict=True)
        print(
            f"[Checkpoint] loaded {len(matched_state)} tensors, "
            f"skipped {skipped_missing_or_mismatch} by key/shape mismatch, "
            f"missing_in_ckpt={len(load_result.missing_keys)}, "
            f"unexpected_in_ckpt={len(load_result.unexpected_keys)}"
        )
        self.model.fuse_inference_kernels()

        self.n_fft = int(self.h.n_fft)
        self.hop_size = int(self.h.hop_size)
        self.win_size = int(self.h.win_size)
        self.compress_factor = float(self.h.compress_factor)
        self.sampling_rate = int(self.h.sampling_rate)
        self.window = torch.hann_window(self.win_size, device=self.device)
        self._graph_ready = False
        self._graph = None
        self._graph_static_amp = None
        self._graph_static_pha = None
        self._graph_in_caches = None
        self._graph_cache_count = 0
        self._graph_out_amp = None
        self._graph_out_pha = None

    def _init_cuda_graph_state(self, batch_size: int, n_freqs: int, dtype: torch.dtype):
        if not self.use_cuda_graph:
            return
        if self._graph_ready:
            static_amp = self._graph_static_amp
            if static_amp is not None and static_amp.shape[0] == batch_size and static_amp.shape[1] == n_freqs and static_amp.dtype == dtype:
                return

        self._graph_static_amp = torch.zeros(batch_size, n_freqs, 1, device=self.device, dtype=dtype)
        self._graph_static_pha = torch.zeros(batch_size, n_freqs, 1, device=self.device, dtype=dtype)
        self._graph_in_caches = self.model.init_stream_cache(batch_size, n_freqs, device=self.device, dtype=dtype)
        self._graph_cache_count = len(self._graph_in_caches)

        warmup_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                step_out = self.model(self._graph_static_amp, self._graph_static_pha, *self._graph_in_caches)
                for cache_in, cache_out in zip(self._graph_in_caches, step_out[3:3 + self._graph_cache_count]):
                    cache_in.copy_(cache_out)
                self._graph_out_amp = step_out[0]
                self._graph_out_pha = step_out[1]
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        self._sync_device()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            step_out = self.model(self._graph_static_amp, self._graph_static_pha, *self._graph_in_caches)
            for cache_in, cache_out in zip(self._graph_in_caches, step_out[3:3 + self._graph_cache_count]):
                cache_in.copy_(cache_out)
            self._graph_out_amp = step_out[0]
            self._graph_out_pha = step_out[1]
        self._graph_ready = True

    def _graph_step(self, amp_t: torch.Tensor, pha_t: torch.Tensor):
        self._graph_static_amp.copy_(amp_t)
        self._graph_static_pha.copy_(pha_t)
        self._graph.replay()
        return self._graph_out_amp, self._graph_out_pha

    def _stft_mag_pha(self, wav_tensor: torch.Tensor):
        spec = torch.stft(
            wav_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        mag = torch.abs(spec).clamp_min(1e-9).pow(self.compress_factor)
        pha = torch.angle(spec)
        return mag, pha

    def _istft_mag_pha(self, mag: torch.Tensor, pha: torch.Tensor):
        mag = mag.clamp_min(1e-9).pow(1.0 / self.compress_factor)
        spec = torch.polar(mag, pha)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
        )
        return wav

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def enhance_waveform_stream(self, noisy_wav: np.ndarray) -> np.ndarray:
        if noisy_wav.ndim != 1:
            raise ValueError("noisy_wav must be 1-D mono waveform")

        noisy = torch.from_numpy(noisy_wav.astype(np.float32)).to(self.device).unsqueeze(0)
        denom = torch.sum(noisy ** 2.0).clamp_min(1e-12)
        norm = 1 # torch.sqrt(torch.tensor(noisy.shape[-1], device=self.device, dtype=noisy.dtype) / denom)
        noisy = noisy * norm

        noisy_amp, noisy_pha = self._stft_mag_pha(noisy)

        b, f, t = noisy_amp.shape
        caches = self.model.init_stream_cache(b, f, device=self.device, dtype=noisy_amp.dtype)
        if self.use_cuda_graph:
            self._init_cuda_graph_state(batch_size=b, n_freqs=f, dtype=noisy_amp.dtype)
            for dst, src in zip(self._graph_in_caches, caches):
                dst.copy_(src)

        out_amp = torch.empty_like(noisy_amp)
        out_pha = torch.empty_like(noisy_pha)

        for i in range(t):
            if self.use_cuda_graph:
                denoised_amp_t, denoised_pha_t = self._graph_step(noisy_amp[:, :, i : i + 1], noisy_pha[:, :, i : i + 1])
                out_amp[:, :, i : i + 1] = denoised_amp_t
                out_pha[:, :, i : i + 1] = denoised_pha_t
            else:
                step_out = self.model(noisy_amp[:, :, i : i + 1], noisy_pha[:, :, i : i + 1], *caches)
                out_amp[:, :, i : i + 1] = step_out[0]
                out_pha[:, :, i : i + 1] = step_out[1]
                caches = step_out[3:]

        enhanced = self._istft_mag_pha(out_amp, out_pha)
        enhanced = enhanced / (norm + 1e-12)
        return enhanced.squeeze(0).detach().cpu().numpy()

    @torch.inference_mode()
    def enhance_file(self, input_wav: str, output_wav: str):
        wav, sr = sf.read(input_wav, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != self.sampling_rate:
            raise ValueError(f"Sample rate mismatch: got {sr}, expected {self.sampling_rate}")

        self._sync_device()
        t0 = time.perf_counter()
        enhanced = self.enhance_waveform_stream(wav)
        self._sync_device()
        t1 = time.perf_counter()

        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, enhanced, self.sampling_rate)
        elapsed = max(t1 - t0, 1e-12)
        duration = max(len(wav) / float(self.sampling_rate), 1e-12)
        return {
            "elapsed_sec": elapsed,
            "audio_sec": duration,
            "rtf": elapsed / duration,
        }
