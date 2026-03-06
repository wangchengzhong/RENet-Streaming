import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None
import scipy.signal as signal


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _trt_dtype_to_torch(dtype):
    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.INT32:
        return torch.int32
    if dtype == trt.DataType.INT8:
        return torch.int8
    raise ValueError(f"Unsupported TRT dtype: {dtype}")


def _shape_tuple(shape):
    return tuple(int(x) for x in shape)


def _build_engine(engine_file):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine_bytes = Path(engine_file).read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_file}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TRT execution context")
    return engine, context


def _collect_io_names(engine):
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    return input_names, output_names


def _alloc_tensor(shape, dtype, device):
    if any(int(x) < 0 for x in shape):
        raise RuntimeError(f"Dynamic shape unresolved: {shape}")
    return torch.zeros(_shape_tuple(shape), dtype=dtype, device=device)


def _is_valid_input_device(devices, idx):
    return isinstance(idx, int) and 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0


def _is_valid_output_device(devices, idx):
    return isinstance(idx, int) and 0 <= idx < len(devices) and devices[idx]["max_output_channels"] > 0


def _normalize_hostapi_name(name: str):
    key = str(name).strip().lower()
    mapping = {
        "auto": None,
        "mme": "mme",
        "ds": "windows directsound",
        "directsound": "windows directsound",
        "wasapi": "windows wasapi",
        "wdm": "windows wdm-ks",
        "wdm-ks": "windows wdm-ks",
        "wdmks": "windows wdm-ks",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported hostapi: {name}")
    return mapping[key]


def _resolve_audio_devices(input_device, output_device, hostapi="auto"):
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    hostapi_filter = _normalize_hostapi_name(hostapi)

    def device_hostapi_name(idx):
        return str(hostapis[devices[idx]["hostapi"]]["name"]).lower()

    def hostapi_ok(idx):
        if hostapi_filter is None:
            return True
        return device_hostapi_name(idx) == hostapi_filter

    default_pair = sd.default.device
    default_in = int(default_pair[0]) if default_pair is not None else -1
    default_out = int(default_pair[1]) if default_pair is not None else -1

    resolved_in = input_device
    resolved_out = output_device

    if resolved_in is not None and not hostapi_ok(int(resolved_in)):
        raise RuntimeError(f"input_device={resolved_in} does not belong to requested hostapi={hostapi}")
    if resolved_out is not None and not hostapi_ok(int(resolved_out)):
        raise RuntimeError(f"output_device={resolved_out} does not belong to requested hostapi={hostapi}")

    if resolved_in is None:
        if _is_valid_input_device(devices, default_in) and hostapi_ok(default_in):
            resolved_in = default_in
        else:
            resolved_in = next((i for i, d in enumerate(devices) if d["max_input_channels"] > 0 and hostapi_ok(i)), None)

    if resolved_out is None:
        if _is_valid_output_device(devices, default_out) and hostapi_ok(default_out):
            resolved_out = default_out
        else:
            resolved_out = next((i for i, d in enumerate(devices) if d["max_output_channels"] > 0 and hostapi_ok(i)), None)

    if resolved_in is None:
        raise RuntimeError("No valid input audio device found")
    if resolved_out is None:
        raise RuntimeError("No valid output audio device found")

    if not _is_valid_input_device(devices, resolved_in):
        raise RuntimeError(f"Selected input_device={resolved_in} is invalid or has no input channels")
    if not _is_valid_output_device(devices, resolved_out):
        raise RuntimeError(f"Selected output_device={resolved_out} is invalid or has no output channels")

    in_api_name = hostapis[devices[resolved_in]["hostapi"]]["name"]
    out_api_name = hostapis[devices[resolved_out]["hostapi"]]["name"]
    if hostapi_filter is not None and str(in_api_name).lower() != str(out_api_name).lower():
        raise RuntimeError(
            f"Resolved devices are not on same hostapi under filter={hostapi}: in={in_api_name}, out={out_api_name}"
        )

    return (
        resolved_in,
        resolved_out,
        devices[resolved_in]["name"],
        devices[resolved_out]["name"],
        in_api_name,
        out_api_name,
    )


def _validate_device_settings(input_device, output_device, samplerate):
    try:
        sd.check_input_settings(device=input_device, samplerate=samplerate, channels=1, dtype="float32")
    except Exception as exc:
        raise RuntimeError(
            f"Input device {input_device} does not support samplerate={samplerate}Hz, channels=1, dtype=float32: {exc}"
        ) from exc

    try:
        sd.check_output_settings(device=output_device, samplerate=samplerate, channels=1, dtype="float32")
    except Exception as exc:
        raise RuntimeError(
            f"Output device {output_device} does not support samplerate={samplerate}Hz, channels=1, dtype=float32: {exc}"
        ) from exc


class DynamicLinearResampler:
    def __init__(self, in_sr: int, out_sr: int):
        self.in_sr = float(in_sr)
        self.out_sr = float(out_sr)
        if self.in_sr <= 0 or self.out_sr <= 0:
            raise ValueError("Sample rates must be positive")
        self.base_step = self.in_sr / self.out_sr
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0.0

    def process(self, x: np.ndarray, speed_ratio: float = 1.0) -> np.ndarray:
        if x is None or len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        self.buffer = np.concatenate((self.buffer, x), axis=0)

        speed_ratio = float(speed_ratio)
        if speed_ratio <= 0:
            speed_ratio = 1.0
        current_step = self.base_step * speed_ratio

        n = self.buffer.shape[0]
        if self.pos + 1.0 >= n:
            return np.zeros(0, dtype=np.float32)

        # Vectorized indexing
        max_idx = int((n - 1.0 - self.pos) / current_step)
        if max_idx < 0:
            return np.zeros(0, dtype=np.float32)

        indices = self.pos + np.arange(max_idx + 1) * current_step
        idx = indices.astype(np.int32)
        frac = indices - idx

        out = self.buffer[idx] * (1.0 - frac) + self.buffer[idx + 1] * frac
        self.pos = indices[-1] + current_step

        drop = int(self.pos)
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.pos -= drop

        return out.astype(np.float32)

class AntiAliasCubicResampler:
    def __init__(self, in_sr: int, out_sr: int):
        self.in_sr = float(in_sr)
        self.out_sr = float(out_sr)
        if self.in_sr <= 0 or self.out_sr <= 0:
            raise ValueError("Sample rates must be positive")
        self.base_step = self.in_sr / self.out_sr
        
        self.buffer = np.zeros(3, dtype=np.float32)
        self.pos = 1.0  

        # Determine direction for filtering
        self.is_downsampling = self.in_sr > self.out_sr
        self.is_upsampling = self.out_sr > self.in_sr

        # Design a stateful Butterworth low-pass filter at 90% of the Nyquist limit
        if self.is_downsampling or self.is_upsampling:
            nyq = min(self.in_sr, self.out_sr) / 2.0
            cutoff = nyq * 0.9 
            fs = max(self.in_sr, self.out_sr)
            
            self.sos = signal.butter(4, cutoff, btype='low', fs=fs, output='sos')
            self.zi = signal.sosfilt_zi(self.sos)

    def process(self, x: np.ndarray, speed_ratio: float = 1.0) -> np.ndarray:
        if x is None or len(x) == 0:
            return np.zeros(0, dtype=np.float32)
            
        x = np.asarray(x, dtype=np.float32)

        # 1. Anti-Aliasing BEFORE downsampling
        if self.is_downsampling:
            x, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
            x = x.astype(np.float32)

        self.buffer = np.concatenate((self.buffer, x), axis=0)

        speed_ratio = float(speed_ratio)
        if speed_ratio <= 0:
            speed_ratio = 1.0
        current_step = self.base_step * speed_ratio

        n = self.buffer.shape[0]
        if self.pos + 2.0 >= n:
            return np.zeros(0, dtype=np.float32)

        max_float_idx = (n - 2.0001 - self.pos) / current_step
        if max_float_idx < 0:
            return np.zeros(0, dtype=np.float32)

        max_idx = int(max_float_idx)
        indices = self.pos + np.arange(max_idx + 1) * current_step
        idx = indices.astype(np.int32)
        frac = indices - idx

        # Cubic Interpolation
        p0 = self.buffer[idx - 1]
        p1 = self.buffer[idx]
        p2 = self.buffer[idx + 1]
        p3 = self.buffer[idx + 2]

        a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
        b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
        c = -0.5 * p0 + 0.5 * p2
        d = p1

        out = a * (frac ** 3) + b * (frac ** 2) + c * frac + d

        self.pos = indices[-1] + current_step

        drop = int(self.pos) - 1
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.pos -= drop

        out = out.astype(np.float32)

        # 2. Anti-Imaging AFTER upsampling
        if self.is_upsampling:
            out, self.zi = signal.sosfilt(self.sos, out, zi=self.zi)
            out = out.astype(np.float32)

        return out
    
class RealtimeTRTEnhancer:
    def __init__(self, engine_file, config_file, use_cuda_graph=True, warmup_frames=8):
        with open(config_file, "r", encoding="utf-8") as f:
            self.h = AttrDict(json.loads(f.read()))

        self.sr = int(self.h.sampling_rate)
        self.n_fft = int(self.h.n_fft)
        self.hop_size = int(self.h.hop_size)
        self.win_size = int(self.h.win_size)
        self.compress_factor = float(self.h.compress_factor)

        if self.hop_size * 2 != self.win_size:
            raise ValueError(
                f"This realtime demo expects 50% overlap (hop*2==win). Got hop={self.hop_size}, win={self.win_size}."
            )
        if self.n_fft != self.win_size:
            raise ValueError(
                f"This realtime demo expects n_fft==win_size for frame OLA. Got n_fft={self.n_fft}, win={self.win_size}."
            )

        self.device = torch.device("cuda")
        self.window = torch.hann_window(self.win_size, device=self.device, dtype=torch.float32)

        self.engine, self.context = _build_engine(engine_file)

        self.context.set_input_shape("noisy_amp_t", (1, self.n_fft // 2 + 1, 1))
        self.context.set_input_shape("noisy_pha_t", (1, self.n_fft // 2 + 1, 1))

        input_names, output_names = _collect_io_names(self.engine)
        self.input_tensors = {}
        self.output_tensors = {}

        for name in input_names:
            shape = self.context.get_tensor_shape(name)
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            self.input_tensors[name] = _alloc_tensor(shape, dtype, self.device)

        for name in output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            self.output_tensors[name] = _alloc_tensor(shape, dtype, self.device)

        self.cache_map = {}
        for name in input_names:
            if name in ["noisy_amp_t", "noisy_pha_t"]:
                continue
            out_name = f"{name}_out"
            if out_name in self.output_tensors:
                self.cache_map[name] = out_name

        for name, tensor in self.input_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        if "denoised_amp_t" not in self.output_tensors or "denoised_pha_t" not in self.output_tensors:
            raise RuntimeError("Engine outputs must include denoised_amp_t and denoised_pha_t")

        self.stream = torch.cuda.Stream(device=self.device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph = None

        self.in_ring = np.zeros(self.win_size, dtype=np.float32)
        self.ola_buffer = np.zeros(self.win_size, dtype=np.float32)
        self.ola_norm_buffer = np.zeros(self.win_size, dtype=np.float32)
        self.window_np = self.window.detach().cpu().numpy().astype(np.float32)
        self.window_sq_np = (self.window_np * self.window_np).astype(np.float32)

        self._warmup(max(1, int(warmup_frames)))
        self._try_capture_graph()

    def _frame_stft(self, frame_np, norm_factor=1.0):
        frame = torch.from_numpy(frame_np).to(self.device)
        
        # Apply the exact same volume scaling as the offline script
        frame = frame * norm_factor
        frame = frame * self.window
        spec = torch.fft.rfft(frame, n=self.n_fft)
        
        real = spec.real
        imag = spec.imag
        
        # Match offline magnitude
        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)
        mag = torch.pow(mag, self.compress_factor)
        
        # Match offline phase bias exactly
        pha = torch.atan2(imag + 1e-10, real + 1e-5)
        
        return mag.view(1, -1, 1), pha.view(1, -1, 1)

    def _frame_istft_ola(self, den_amp_t, den_pha_t):
        mag = den_amp_t.squeeze(0).squeeze(-1)
        pha = den_pha_t.squeeze(0).squeeze(-1)
        mag = torch.pow(torch.clamp(mag, min=1e-9), 1.0 / self.compress_factor)
        mag = torch.clamp(mag, max=10.0)

        spec = torch.polar(mag, pha)
        frame = torch.fft.irfft(spec, n=self.n_fft)[: self.win_size]
        frame = frame * self.window
        frame_np = frame.detach().cpu().numpy().astype(np.float32)

        self.ola_buffer += frame_np
        self.ola_norm_buffer += self.window_sq_np
        out = self.ola_buffer[: self.hop_size] / np.maximum(self.ola_norm_buffer[: self.hop_size], 1e-4)
        out = out.astype(np.float32, copy=True)
        self.ola_buffer[:-self.hop_size] = self.ola_buffer[self.hop_size :]
        self.ola_buffer[-self.hop_size :] = 0.0
        self.ola_norm_buffer[:-self.hop_size] = self.ola_norm_buffer[self.hop_size :]
        self.ola_norm_buffer[-self.hop_size :] = 0.0
        return out

    def _execute_step(self):
        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for cache_in_name, cache_out_name in self.cache_map.items():
            self.input_tensors[cache_in_name].copy_(self.output_tensors[cache_out_name])

    def _warmup(self, warmup_frames):
        zero = (np.random.randn(self.hop_size).astype(np.float32) * 1e-5)
        with torch.cuda.stream(self.stream):
            for _ in range(warmup_frames):
                self.in_ring[:-self.hop_size] = self.in_ring[self.hop_size :]
                self.in_ring[-self.hop_size :] = zero
                amp_t, pha_t = self._frame_stft(self.in_ring)
                self.input_tensors["noisy_amp_t"].copy_(amp_t)
                self.input_tensors["noisy_pha_t"].copy_(pha_t)
                self._execute_step()
        self.stream.synchronize()

    def _try_capture_graph(self):
        if not self.use_cuda_graph:
            return
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=self.stream):
                self._execute_step()
            self.graph = graph
        except Exception as exc:
            self.graph = None
            self.use_cuda_graph = False
            print(f"cuda_graph_capture_failed={exc}")

    # def process_hop(self, in_hop_np, bypass_denoise=False):
    #     self.in_ring[:-self.hop_size] = self.in_ring[self.hop_size :]
    #     self.in_ring[-self.hop_size :] = in_hop_np

    #     amp_t, pha_t = self._frame_stft(self.in_ring)
    #     with torch.cuda.stream(self.stream):
    #         self.input_tensors["noisy_amp_t"].copy_(amp_t)
    #         self.input_tensors["noisy_pha_t"].copy_(pha_t)
    #         if self.graph is not None:
    #             self.graph.replay()
    #         else:
    #             self._execute_step()
    #     self.stream.synchronize()

    #     den_amp_t = self.output_tensors["denoised_amp_t"]
    #     den_pha_t = self.output_tensors["denoised_pha_t"]
    #     if bypass_denoise:
    #         return self._frame_istft_ola(amp_t, pha_t)
    #     return self._frame_istft_ola(den_amp_t, den_pha_t)
    def process_hop(self, in_hop_np, bypass_denoise=False):
        self.in_ring[:-self.hop_size] = self.in_ring[self.hop_size :]
        self.in_ring[-self.hop_size :] = in_hop_np

        # Force all PyTorch DSP and TensorRT inference to wait in a single, unified line
        with torch.cuda.stream(self.stream):
            amp_t, pha_t = self._frame_stft(self.in_ring)
            
            self.input_tensors["noisy_amp_t"].copy_(amp_t)
            self.input_tensors["noisy_pha_t"].copy_(pha_t)
            
            if self.graph is not None:
                self.graph.replay()
            else:
                self._execute_step()
                
            den_amp_t = self.output_tensors["denoised_amp_t"]
            den_pha_t = self.output_tensors["denoised_pha_t"]
            
            if bypass_denoise:
                out_chunk = self._frame_istft_ola(amp_t, pha_t)
            else:
                out_chunk = self._frame_istft_ola(den_amp_t, den_pha_t)
                
        # Synchronize once at the very end before returning to CPU
        self.stream.synchronize()
        return out_chunk

def main():
    parser = argparse.ArgumentParser(description="Realtime microphone demo: TensorRT + optional CUDA Graph + OLA output")
    parser.add_argument(
        "--engine_file",
        default="generated_files/trt_cache/TensorrtExecutionProvider_TRTKernel_graph_main_graph_1772623164867571760_0_0_sm89.engine",
    )
    parser.add_argument("--config_file", default="cp_model_casual_whamr/config.json")
    parser.add_argument("--input_device", type=int, default=None)
    parser.add_argument("--output_device", type=int, default=None)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--use_cuda_graph", action="store_true")
    parser.add_argument("--queue_blocks", type=int, default=96, help="Input/output queue capacity in hop blocks")
    parser.add_argument("--prefill_blocks", type=int, default=12, help="Prefill output blocks before starting playback")
    parser.add_argument("--device_sr", type=int, default=None, help="Audio device sample rate. If omitted, uses model sample rate")
    parser.add_argument("--device_blocksize", type=int, default=0, help="Audio callback blocksize at device_sr. 0 means auto")
    parser.add_argument("--hostapi", default="auto", choices=["auto", "mme", "directsound", "ds", "wasapi", "wdm-ks", "wdm", "wdmks"], help="Prefer/select a host API and keep I/O on same API")
    parser.add_argument("--latency", default="high", help="sounddevice stream latency, e.g. low/high/seconds")
    parser.add_argument("--stats_interval", type=float, default=2.0, help="Seconds between realtime stats prints; <=0 disables")
    parser.add_argument("--drift_kp", type=float, default=0.01, help="Proportional gain for output-queue drift controller")
    parser.add_argument("--drift_max_dev", type=float, default=0.01, help="Max absolute speed ratio deviation for ASRC controller")
    parser.add_argument("--enter_toggle_raw", action="store_true", help="Press Enter to toggle output between denoised and original speech")
    parser.add_argument("--bypass_model", action="store_true", help="Bypass TRT model and pass input hop to output directly (for device-path diagnosis)")
    parser.add_argument("--list_devices", action="store_true")
    args = parser.parse_args()

    if sd is None:
        raise RuntimeError("sounddevice is not installed. Install it first: pip install sounddevice")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.list_devices:
        print(sd.query_devices())
        return

    enhancer = None
    model_sr = None
    model_hop = None
    if not args.bypass_model:
        enhancer = RealtimeTRTEnhancer(
            engine_file=args.engine_file,
            config_file=args.config_file,
            use_cuda_graph=args.use_cuda_graph,
            warmup_frames=args.warmup_frames,
        )
        model_sr = enhancer.sr
        model_hop = enhancer.hop_size
        use_cuda_graph_actual = enhancer.use_cuda_graph
        n_fft = enhancer.n_fft
        win_size = enhancer.win_size
    else:
        with open(args.config_file, "r", encoding="utf-8") as f:
            h = AttrDict(json.loads(f.read()))
        model_sr = int(h.sampling_rate)
        model_hop = int(h.hop_size)
        n_fft = int(h.n_fft)
        win_size = int(h.win_size)
        use_cuda_graph_actual = False

    device_sr = int(args.device_sr) if args.device_sr is not None else int(model_sr)
    if int(args.device_blocksize) > 0:
        device_blocksize = int(args.device_blocksize)
    else:
        device_blocksize = max(1, int(round(device_sr * model_hop / float(model_sr))))

    input_device, output_device, input_name, output_name, input_api, output_api = _resolve_audio_devices(
        args.input_device,
        args.output_device,
        hostapi=args.hostapi,
    )
    _validate_device_settings(input_device, output_device, device_sr)

    print(
        f"Realtime start: model_sr={model_sr}, device_sr={device_sr}, model_hop={model_hop}, device_block={device_blocksize}, n_fft={n_fft}, "
        f"win_size={win_size}, "
        f"use_cuda_graph={use_cuda_graph_actual}, bypass_model={args.bypass_model}"
    )
    print(f"audio_device_in={input_device} ({input_name})")
    print(f"audio_device_out={output_device} ({output_name})")
    print(f"audio_hostapi_in={input_api}, audio_hostapi_out={output_api}")
    print("Press Ctrl+C to stop")

    callback_err_count = {"count": 0}
    dropped_input_blocks = {"count": 0}
    dropped_output_blocks = {"count": 0}
    missed_output_blocks = {"count": 0}
    played_blocks = {"count": 0}
    frame_mismatch_count = {"count": 0}
    frame_mismatch_last = {"value": -1}
    callback_exception_count = {"count": 0}

    qsize = max(4, int(args.queue_blocks))
    prefill_blocks = max(1, min(int(args.prefill_blocks), qsize - 1))
    stop_event = threading.Event()
    worker = None
    monitor = None
    key_toggle_thread = None
    in_queue = None
    out_queue = None
    raw_mode_state = {"enabled": False}
    last_output_value = {"value": 0.0}
    playback_buffer = np.zeros(0, dtype=np.float32)

    if not args.bypass_model:
        in_queue = queue.Queue(maxsize=qsize)
        out_queue = queue.Queue(maxsize=qsize)
        in_resampler = AntiAliasCubicResampler(device_sr, model_sr)
        out_resampler = AntiAliasCubicResampler(model_sr, device_sr)
        model_input_buffer = np.zeros(0, dtype=np.float32)
        target_out_q = max(1, prefill_blocks // 2)
        kp = float(args.drift_kp)
        max_deviation = max(0.0, float(args.drift_max_dev))

        def worker_loop():
            nonlocal model_input_buffer
            ema_fill = float(target_out_q)
            current_speed_ratio = 1.0  # Tracks the smoothed physical speed

            while not stop_event.is_set():
                try:
                    in_block = in_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if in_block is None:
                    break
                model_chunk = in_resampler.process(in_block, speed_ratio=1.0)
                if model_chunk.size > 0:
                    model_input_buffer = np.concatenate((model_input_buffer, model_chunk), axis=0)

                while model_input_buffer.size >= model_hop:
                    hop_in = model_input_buffer[:model_hop]
                    model_input_buffer = model_input_buffer[model_hop:]
                    enhanced_hop = enhancer.process_hop(hop_in, bypass_denoise=raw_mode_state["enabled"])

                    fill = out_queue.qsize()
                    
                    # 1. Ultra-slow EMA (time constant of ~2 seconds)
                    ema_fill = 0.995 * ema_fill + 0.005 * float(fill)
                    
                    # 2. Calculate target based on the smoothed queue size
                    error = ema_fill - target_out_q
                    target_ratio = 1.0 + error * kp
                    target_ratio = max(1.0 - max_deviation, min(1.0 + max_deviation, target_ratio))

                    # 3. Slew-rate limiter: Prevents the 100Hz FM sawtooth artifact
                    # by forcing the speed to change imperceptibly slowly.
                    if target_ratio > current_speed_ratio + 0.00005:
                        current_speed_ratio += 0.00005
                    elif target_ratio < current_speed_ratio - 0.00005:
                        current_speed_ratio -= 0.00005
                    else:
                        current_speed_ratio = target_ratio

                    dev_chunk = out_resampler.process(enhanced_hop, speed_ratio=current_speed_ratio)
                    
                    if dev_chunk.size == 0:
                        continue
                    try:
                        out_queue.put_nowait(dev_chunk)
                    except queue.Full:
                        dropped_output_blocks["count"] += 1

        worker = threading.Thread(target=worker_loop, daemon=True)
        worker.start()

        # def monitor_loop():
        #     interval = float(args.stats_interval)
        #     if interval <= 0:
        #         return
        #     while not stop_event.is_set():
        #         time.sleep(interval)
        #         if stop_event.is_set():
        #             break
        #         print(
        #             "rt_stats "
        #             f"in_q={in_queue.qsize()}/{qsize}, out_q={out_queue.qsize()}/{qsize}, "
        #             f"played={played_blocks['count']}, dropped_in={dropped_input_blocks['count']}, "
        #             f"dropped_out={dropped_output_blocks['count']}, missed_out={missed_output_blocks['count']}, callback_status={callback_err_count['count']}, "
        #             f"frame_mismatch={frame_mismatch_count['count']}"
        #         )

        # monitor = threading.Thread(target=monitor_loop, daemon=True)
        # monitor.start()

    if (not args.bypass_model) and args.enter_toggle_raw:
        print("Enter-toggle enabled: press Enter to switch between DENOISED and ORIGINAL output.")

        def key_toggle_loop():
            while not stop_event.is_set():
                try:
                    _ = input()
                except EOFError:
                    break
                raw_mode_state["enabled"] = not raw_mode_state["enabled"]
                mode = "ORIGINAL" if raw_mode_state["enabled"] else "DENOISED"
                print(f"output_mode={mode}")

        key_toggle_thread = threading.Thread(target=key_toggle_loop, daemon=True)
        key_toggle_thread.start()

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal playback_buffer
        try:
            del time_info
            if status:
                callback_err_count["count"] += 1
                if callback_err_count["count"] <= 5:
                    print(f"audio_status={status}")

            # 1. READ AND SANITIZE INPUT
            mono = indata[:frames, 0].astype(np.float32, copy=True)
            mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
            mono = np.clip(mono, -1.0, 1.0)

            if args.bypass_model:
                outdata[:, 0] = mono
                played_blocks["count"] += 1
                return

            try:
                in_queue.put_nowait(mono.copy())
            except queue.Full:
                dropped_input_blocks["count"] += 1

            while playback_buffer.size < frames:
                try:
                    chunk = out_queue.get_nowait()
                    if chunk.size > 0:
                        playback_buffer = np.concatenate((playback_buffer, chunk), axis=0)
                except queue.Empty:
                    break

            outdata.fill(0.0)
            n_out = min(frames, playback_buffer.size)
            
            if n_out > 0:
                # 2. SANITIZE AND CLAMP OUTPUT
                safe_out = np.nan_to_num(playback_buffer[:n_out], nan=0.0, posinf=1.0, neginf=-1.0)
                clamped_out = np.clip(safe_out, -1.0, 1.0)
                
                outdata[:n_out, 0] = clamped_out
                last_output_value["value"] = float(clamped_out[-1])
                playback_buffer = playback_buffer[n_out:]
                
            if n_out < frames:
                missed_output_blocks["count"] += 1
                outdata[n_out:frames, 0] = 0.0
                
            played_blocks["count"] += 1
            
        except Exception as exc:
            callback_exception_count["count"] += 1
            if callback_exception_count["count"] <= 5:
                print(f"audio_callback_exception={type(exc).__name__}: {exc}")
            outdata.fill(0.0)

    stream = sd.Stream(
        samplerate=device_sr,
        blocksize=device_blocksize,
        dtype="float32",
        channels=1,
        callback=audio_callback,
        device=(input_device, output_device),
        latency=args.latency,
    )

    print(f"queue_blocks={qsize}, prefill_blocks={prefill_blocks}, latency={args.latency}")
    if not args.bypass_model:
        silent = np.zeros(device_blocksize, dtype=np.float32)
        for _ in range(prefill_blocks):
            try:
                in_queue.put_nowait(silent.copy())
            except queue.Full:
                break
        t_wait = time.time() + 1.0
        while out_queue.qsize() < max(1, prefill_blocks // 2) and time.time() < t_wait:
            time.sleep(0.01)

    with stream:
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    stop_event.set()
    if worker is not None and in_queue is not None:
        try:
            in_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=1.0)
    if monitor is not None:
        monitor.join(timeout=0.5)
    if key_toggle_thread is not None:
        key_toggle_thread.join(timeout=0.2)

    print(f"audio_callback_status_events={callback_err_count['count']}")
    print(f"audio_callback_exceptions={callback_exception_count['count']}")
    print(f"dropped_input_blocks={dropped_input_blocks['count']}")
    print(f"dropped_output_blocks={dropped_output_blocks['count']}")
    print(f"missed_output_blocks={missed_output_blocks['count']}")
    print(f"frame_mismatch_count={frame_mismatch_count['count']}, last_frames={frame_mismatch_last['value']}")
    print("Realtime stopped")


if __name__ == "__main__":
    main()
