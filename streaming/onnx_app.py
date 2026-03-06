import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=0.3, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

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


def _resolve_dim(dim, batch_size, n_freqs):
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        key = dim.lower()
        if "batch" in key:
            return batch_size
        if "freq" in key:
            return n_freqs
    if dim is None:
        return 1
    return 1


def _build_cache_from_onnx_inputs(session, batch_size, n_freqs, dtype=np.float32):
    caches = []
    input_names = [x.name for x in session.get_inputs()]
    for inp in session.get_inputs()[2:]:
        shape = [_resolve_dim(d, batch_size, n_freqs) for d in inp.shape]
        caches.append(np.zeros(shape, dtype=dtype))
    return input_names, caches


def _build_gpu_buffers_for_iobinding(session, batch_size, n_freqs, device):
    input_buffers = {}
    output_buffers = {}
    for inp in session.get_inputs():
        shape = [_resolve_dim(d, batch_size, n_freqs) for d in inp.shape]
        input_buffers[inp.name] = torch.zeros(shape, device=device, dtype=torch.float32)
    for out in session.get_outputs():
        shape = [_resolve_dim(d, batch_size, n_freqs) for d in out.shape]
        output_buffers[out.name] = torch.zeros(shape, device=device, dtype=torch.float32)
    return input_buffers, output_buffers


def _bind_iobinding(session, io_binding, input_buffers, output_buffers, device_id):
    for name, tensor in input_buffers.items():
        io_binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=list(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
    for name, tensor in output_buffers.items():
        io_binding.bind_output(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=list(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )


def _prepare_pingpong_iobinding(session, batch_size, n_freqs, device, device_id=0):
    input_metas = {x.name: x for x in session.get_inputs()}
    output_metas = {x.name: x for x in session.get_outputs()}
    out_names = [x.name for x in session.get_outputs()]

    cache_in_names = [x.name for x in session.get_inputs()[2:]]
    cache_out_names = [x.name for x in session.get_outputs()[3:]]

    frame_inputs = {
        "noisy_amp_t": torch.zeros(
            [_resolve_dim(d, batch_size, n_freqs) for d in input_metas["noisy_amp_t"].shape],
            device=device,
            dtype=torch.float32,
        ),
        "noisy_pha_t": torch.zeros(
            [_resolve_dim(d, batch_size, n_freqs) for d in input_metas["noisy_pha_t"].shape],
            device=device,
            dtype=torch.float32,
        ),
    }

    cache_banks = []
    for _ in range(2):
        bank = {}
        for name in cache_in_names:
            shape = [_resolve_dim(d, batch_size, n_freqs) for d in input_metas[name].shape]
            bank[name] = torch.zeros(shape, device=device, dtype=torch.float32)
        cache_banks.append(bank)

    bindings = []
    for in_bank_idx in [0, 1]:
        out_bank_idx = 1 - in_bank_idx
        input_buffers = {**frame_inputs, **cache_banks[in_bank_idx]}
        output_buffers = {}

        for name in out_names[:3]:
            shape = [_resolve_dim(d, batch_size, n_freqs) for d in output_metas[name].shape]
            output_buffers[name] = torch.zeros(shape, device=device, dtype=torch.float32)

        for in_name, out_name in zip(cache_in_names, cache_out_names):
            output_buffers[out_name] = cache_banks[out_bank_idx][in_name]

        io_binding = session.io_binding()
        _bind_iobinding(session, io_binding, input_buffers, output_buffers, device_id=device_id)
        bindings.append({"io_binding": io_binding, "output_buffers": output_buffers})

    return frame_inputs, bindings, out_names


def main():
    parser = argparse.ArgumentParser(description="ONNXRuntime-GPU FP32 streaming benchmark")
    parser.add_argument("--onnx_file", default="cp_model_casual_whamr/stream.onnx")
    parser.add_argument("--config_file", default="cp_model_casual_whamr/config.json")
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--provider", default="trt", choices=["trt", "cuda", "cpu"])
    parser.add_argument("--output_wav", default="wavs/enh_onnx_stream.wav")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--trt_fp16", action="store_true")
    parser.add_argument("--trt_cache_dir", default="generated_files/trt_cache")
    parser.add_argument("--use_iobinding", action="store_true", help="Use ORT IOBinding with GPU-resident buffers (CUDA provider only)")
    parser.add_argument("--ort_log_severity", type=int, default=3, choices=[0, 1, 2, 3, 4], help="ORT log severity: 0=verbose,1=info,2=warning,3=error,4=fatal")
    parser.add_argument("--strict_cuda_ep", action="store_true", help="CUDA provider only (no CPU fallback) to detect/avoid cross-EP memcpy")
    parser.add_argument("--strict_trt_ep", action="store_true", help="TRT provider only (no CUDA/CPU fallback) to detect/avoid cross-EP memcpy")
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        h = AttrDict(json.loads(f.read()))

    wav, sr = sf.read(args.input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != h.sampling_rate:
        raise ValueError(f"sr mismatch: {sr} vs {h.sampling_rate}")
    # wav = wav[: int(args.seconds * sr)]

    noisy = torch.from_numpy(wav).float().unsqueeze(0)
    norm = 1 # torch.sqrt(torch.tensor(noisy.shape[-1], dtype=noisy.dtype) / (torch.sum(noisy ** 2.0) + 1e-12))
    noisy = noisy * norm
    noisy_amp, noisy_pha, _ = mag_pha_stft(noisy, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = int(args.ort_log_severity)
    if args.provider == "trt":
        trt_provider = (
            "TensorrtExecutionProvider",
            {
                "trt_engine_cache_enable": "True",
                "trt_max_workspace_size": "4294967296",
                "trt_engine_cache_path": args.trt_cache_dir,
                "trt_fp16_enable": "True" if args.trt_fp16 else "False",
                "trt_builder_optimization_level": "5",
                "trt_auxiliary_streams": "10"
            },
        )
        cuda_provider = (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            },
        )
        providers = [trt_provider] if args.strict_trt_ep else [trt_provider, cuda_provider, "CPUExecutionProvider"]
    elif args.provider == "cuda":
        cuda_provider = (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "enable_cuda_graph": "0",
            },
        )
        providers = [cuda_provider] if args.strict_cuda_ep else [cuda_provider, "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.onnx_file, sess_options=so, providers=providers)
    out_names = [x.name for x in sess.get_outputs()]

    use_iobinding = bool(args.use_iobinding and args.provider in ["cuda", "trt"])
    T = noisy_amp.shape[-1]
    warmup_frames = max(1, min(args.warmup_frames, T))

    if use_iobinding:
        device = torch.device("cuda")
        noisy_amp_gpu = noisy_amp.to(device=device, dtype=torch.float32, non_blocking=True)
        noisy_pha_gpu = noisy_pha.to(device=device, dtype=torch.float32, non_blocking=True)
        enhanced_amp_gpu = torch.empty_like(noisy_amp_gpu)
        enhanced_pha_gpu = torch.empty_like(noisy_pha_gpu)
        frame_inputs, pingpong_bindings, out_names = _prepare_pingpong_iobinding(
            sess,
            batch_size=1,
            n_freqs=noisy_amp.shape[1],
            device=device,
        )

        step_counter = 0

        def run_step(frame_idx, collect_output=False):
            nonlocal step_counter
            binding_state = pingpong_bindings[step_counter & 1]
            frame_inputs["noisy_amp_t"].copy_(noisy_amp_gpu[:, :, frame_idx : frame_idx + 1])
            frame_inputs["noisy_pha_t"].copy_(noisy_pha_gpu[:, :, frame_idx : frame_idx + 1])
            sess.run_with_iobinding(binding_state["io_binding"])
            if collect_output:
                enhanced_amp_gpu[:, :, frame_idx : frame_idx + 1].copy_(binding_state["output_buffers"][out_names[0]])
                enhanced_pha_gpu[:, :, frame_idx : frame_idx + 1].copy_(binding_state["output_buffers"][out_names[1]])
            step_counter += 1

        for i in range(warmup_frames):
            run_step(i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(T):
            run_step(i, collect_output=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        enhanced_amp = enhanced_amp_gpu
        enhanced_pha = enhanced_pha_gpu
    else:
        in_names, caches = _build_cache_from_onnx_inputs(sess, batch_size=1, n_freqs=noisy_amp.shape[1], dtype=np.float32)
        noisy_amp_np = noisy_amp.numpy().astype(np.float32, copy=False)
        noisy_pha_np = noisy_pha.numpy().astype(np.float32, copy=False)
        enhanced_amp_np = np.empty_like(noisy_amp_np)
        enhanced_pha_np = np.empty_like(noisy_pha_np)

        amp0 = noisy_amp_np[:, :, 0:1]
        pha0 = noisy_pha_np[:, :, 0:1]
        feed = {"noisy_amp_t": amp0, "noisy_pha_t": pha0}
        for n, c in zip(in_names[2:], caches):
            feed[n] = c

        for i in range(warmup_frames):
            feed["noisy_amp_t"] = noisy_amp_np[:, :, i : i + 1]
            feed["noisy_pha_t"] = noisy_pha_np[:, :, i : i + 1]
            outs = sess.run(out_names, feed)
            caches = outs[3:]
            for n, c in zip(in_names[2:], caches):
                feed[n] = c

        t0 = time.perf_counter()
        for i in range(T):
            feed["noisy_amp_t"] = noisy_amp_np[:, :, i : i + 1]
            feed["noisy_pha_t"] = noisy_pha_np[:, :, i : i + 1]
            outs = sess.run(out_names, feed)
            enhanced_amp_np[:, :, i : i + 1] = outs[0]
            enhanced_pha_np[:, :, i : i + 1] = outs[1]
            caches = outs[3:]
            for n, c in zip(in_names[2:], caches):
                feed[n] = c
        t1 = time.perf_counter()

        enhanced_amp = torch.from_numpy(enhanced_amp_np)
        enhanced_pha = torch.from_numpy(enhanced_pha_np)

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
    enhanced = enhanced # / (norm.to(enhanced.device) + 1e-12)
    enhanced_np = enhanced.squeeze(0).detach().cpu().numpy()

    Path(args.output_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, enhanced_np, sr)

    print(f"providers={sess.get_providers()}")
    print(f"provider_mode={args.provider}, warmup_frames={warmup_frames}, trt_fp16={args.trt_fp16}, iobinding={use_iobinding}, ort_cuda_graph=False, strict_cuda_ep={args.strict_cuda_ep}, strict_trt_ep={args.strict_trt_ep}, ort_log_severity={args.ort_log_severity}")
    print(f"audio_sec={audio_sec:.3f}, elapsed={elapsed:.3f}, rtf={rtf:.4f}")
    print(f"saved_enhanced={args.output_wav}")


if __name__ == "__main__":
    main()
