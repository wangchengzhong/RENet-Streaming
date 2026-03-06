import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorrt as trt
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=0.3, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=0.3, center=True):
    hann_window = torch.hann_window(win_size).to(mag.device)
    mag = torch.pow(torch.clamp(mag, min=1e-9), 1.0 / compress_factor)
    spec = torch.polar(mag, pha)
    wav = torch.istft(
        spec,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )
    return wav


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


def _set_input_shapes(context, n_freqs):
    context.set_input_shape("noisy_amp_t", (1, n_freqs, 1))
    context.set_input_shape("noisy_pha_t", (1, n_freqs, 1))


def _alloc_tensor(shape, dtype, device):
    if any(int(x) < 0 for x in shape):
        raise RuntimeError(f"Dynamic shape unresolved: {shape}")
    return torch.zeros(_shape_tuple(shape), dtype=dtype, device=device)


def _prepare_bindings(engine, context, device):
    input_names, output_names = _collect_io_names(engine)

    cache_input_names = [n for n in input_names if n not in ["noisy_amp_t", "noisy_pha_t"]]
    cache_output_names = [n for n in output_names if n.endswith("_out")]

    input_tensors = {}
    output_tensors = {}

    for name in input_names:
        shape = context.get_tensor_shape(name)
        dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
        input_tensors[name] = _alloc_tensor(shape, dtype, device)

    for name in output_names:
        shape = context.get_tensor_shape(name)
        dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
        output_tensors[name] = _alloc_tensor(shape, dtype, device)

    cache_map = {}
    for in_name in cache_input_names:
        out_name = f"{in_name}_out"
        if out_name in output_tensors:
            cache_map[in_name] = out_name

    for name, tensor in input_tensors.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))
    for name, tensor in output_tensors.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))

    return input_names, output_names, input_tensors, output_tensors, cache_map


def main():
    parser = argparse.ArgumentParser(description="TensorRT runtime streaming app with optional CUDA Graph replay")
    parser.add_argument(
        "--engine_file",
        default="generated_files/trt_cache/TensorrtExecutionProvider_TRTKernel_graph_main_graph_1772623164867571760_0_0_sm89.engine",
    )
    parser.add_argument("--config_file", default="cp_model_casual_whamr/config.json")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="wavs/enh_trt_cuda_graph.wav")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--use_cuda_graph", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with open(args.config_file, "r", encoding="utf-8") as f:
        h = AttrDict(json.loads(f.read()))

    wav, sr = sf.read(args.input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != h.sampling_rate:
        raise ValueError(f"sr mismatch: {sr} vs {h.sampling_rate}")

    device = torch.device("cuda")
    noisy = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    norm = 1 # torch.sqrt(torch.tensor(noisy.shape[-1], device=device, dtype=noisy.dtype) / (torch.sum(noisy ** 2.0) + 1e-12))
    noisy = noisy * norm
    noisy_amp, noisy_pha, _ = mag_pha_stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    engine, context = _build_engine(args.engine_file)
    n_freqs = noisy_amp.shape[1]
    _set_input_shapes(context, n_freqs)

    (
        input_names,
        output_names,
        input_tensors,
        output_tensors,
        cache_map,
    ) = _prepare_bindings(engine, context, device)

    if "denoised_amp_t" not in output_tensors or "denoised_pha_t" not in output_tensors:
        raise RuntimeError("Engine outputs do not include denoised_amp_t / denoised_pha_t")

    T = noisy_amp.shape[-1]
    warmup_frames = max(1, min(args.warmup_frames, T))

    enhanced_amp = torch.empty_like(noisy_amp)
    enhanced_pha = torch.empty_like(noisy_pha)

    static_amp_t = input_tensors["noisy_amp_t"]
    static_pha_t = input_tensors["noisy_pha_t"]

    stream = torch.cuda.Stream(device=device)

    def one_step(frame_idx, collect_output=False):
        static_amp_t.copy_(noisy_amp[:, :, frame_idx : frame_idx + 1])
        static_pha_t.copy_(noisy_pha[:, :, frame_idx : frame_idx + 1])
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for cache_in_name, cache_out_name in cache_map.items():
            input_tensors[cache_in_name].copy_(output_tensors[cache_out_name])
        if collect_output:
            enhanced_amp[:, :, frame_idx : frame_idx + 1].copy_(output_tensors["denoised_amp_t"])
            enhanced_pha[:, :, frame_idx : frame_idx + 1].copy_(output_tensors["denoised_pha_t"])

    with torch.cuda.stream(stream):
        for i in range(warmup_frames):
            one_step(i, collect_output=False)
    stream.synchronize()

    use_graph = bool(args.use_cuda_graph)
    graph = None
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                ok = context.execute_async_v3(stream.cuda_stream)
                if not ok:
                    raise RuntimeError("TensorRT execute_async_v3 failed during capture")
                for cache_in_name, cache_out_name in cache_map.items():
                    input_tensors[cache_in_name].copy_(output_tensors[cache_out_name])
        except Exception as exc:
            graph = None
            use_graph = False
            print(f"cuda_graph_capture_failed={exc}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if use_graph and graph is not None:
        with torch.cuda.stream(stream):
            for i in range(T):
                static_amp_t.copy_(noisy_amp[:, :, i : i + 1])
                static_pha_t.copy_(noisy_pha[:, :, i : i + 1])
                graph.replay()
                enhanced_amp[:, :, i : i + 1].copy_(output_tensors["denoised_amp_t"])
                enhanced_pha[:, :, i : i + 1].copy_(output_tensors["denoised_pha_t"])
    else:
        with torch.cuda.stream(stream):
            for i in range(T):
                one_step(i, collect_output=True)

    stream.synchronize()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    audio_sec = len(wav) / float(sr)
    rtf = elapsed / max(audio_sec, 1e-12)

    enhanced = mag_pha_istft(
        enhanced_amp,
        enhanced_pha,
        h.n_fft,
        h.hop_size,
        h.win_size,
        h.compress_factor,
    )
    enhanced = enhanced / (norm + 1e-12)
    enhanced_np = enhanced.squeeze(0).detach().cpu().numpy()

    Path(args.output_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, enhanced_np, sr)

    print(f"engine={args.engine_file}")
    print(f"use_cuda_graph={use_graph}")
    print(f"audio_sec={audio_sec:.3f}, elapsed={elapsed:.3f}, rtf={rtf:.4f}")
    print(f"saved_enhanced={args.output_wav}")


if __name__ == "__main__":
    main()
