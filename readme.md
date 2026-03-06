# RENet-Streaming

Streaming and deployment-oriented implementation of **RENet** for real-time speech enhancement.

## Demo

https://github.com/user-attachments/assets/31b934b1-6d13-4408-9424-1639b5ad0131

This repository focuses on the **frame-by-frame inference path** of RENet, including:

- PyTorch streaming inference
- ONNX export for the streaming model
- ONNX Runtime execution with CUDA / TensorRT providers
- native TensorRT runtime inference
- real-time microphone demo on Windows

The original non-streaming research repository is available at:

<https://github.com/wangchengzhong/RENet>

## Background

RENet-Streaming is based on the paper:

**Global Rotation Equivariant Phase Modeling for Speech Enhancement with Deep Magnitude-Phase Interaction**

The original project studies a magnitude-phase dual-stream speech enhancement model with:

- **GRE**: global rotation equivariance in the phase branch
- **MPICM**: magnitude-phase interaction modules
- **HADF**: hybrid attention dual-FFN design

This repository is a practical deployment branch centered on **streaming inference**, rather than the full training / evaluation pipeline from the original project.

## What is included

### Core model code

- `models/model.py`: streaming RENet model definitions, cache logic, causal normalization, inference fusion helpers
- `models/transformer.py`: transformer / FFN blocks used by the model, including causal streaming code paths

### Deployment and validation tools

- `streaming/engine.py`: PyTorch streaming inference engine for waveform input/output
- `streaming/demo_cuda_graph.py`: file-based PyTorch demo with optional CUDA Graph replay
- `export_stream_onnx.py`: export the streaming model to ONNX
- `streaming/onnx_app.py`: run streaming inference with ONNX Runtime on CPU / CUDA / TensorRT EP
- `streaming/trt_cuda_graph_app.py`: run a serialized TensorRT engine on an input wav
- `streaming/realtime_trt_demo.py`: microphone-to-speaker real-time demo using TensorRT
- `check_offline_vs_stream.py`: compare offline `MPNet` and streaming `StreamMPNet` outputs for consistency

### Assets in this repository

- `cp_model_casual_whamr/`: checkpoint/config directory used by the demos
- `generated_files/`: generated TensorRT cache / profiling artifacts
- `wavs/`: sample audio inputs / outputs




### Minimal PyTorch path

```bash
pip install torch numpy soundfile joblib pesq
```

### ONNX Runtime path

```bash
pip install onnxruntime-gpu
```

### Realtime / TensorRT path

```bash
pip install scipy sounddevice
```

Before using any TensorRT-related command in this repository, **install TensorRT first**.

TensorRT Python bindings must be installed separately from NVIDIA packages matching your CUDA / driver environment. ONNX Runtime TensorRT EP also depends on a working TensorRT installation.

In practice, the recommended order is:

1. install CUDA-compatible PyTorch
2. install TensorRT
3. install `onnxruntime-gpu`
4. install optional realtime packages such as `sounddevice` and `scipy`

## Streaming configuration note



## Checkpoints

If you want to **train** a model for this streaming branch, the intended workflow is:

1. start from the original RENet repository: <https://github.com/wangchengzhong/RENet>
2. For practical deployment on a custom PC, the hop size should be increased from 100 to 200 to obtain an available real-time factor (RTF). Make sure the training/export config is consistent with `hop_size = 200`.
3. replace its `models/model.py` and `models/transformer.py` with the versions from this repository
4. train in the original RENet training pipeline
5. obtain the trained checkpoint
6. use that trained checkpoint here for ONNX export and deployment

This repository mainly provides the **streaming model implementation and deployment path**, while training is expected to be done on top of the original RENet codebase after swapping in the streaming-aware model files.

The scripts assume a checkpoint directory like:

```text
cp_model_casual_whamr/
├─ config.json
├─ g_00120000
└─ ...
```

Notes:

- `config.json` is loaded automatically from the checkpoint folder in most scripts.
- the ONNX export script loads weights from `checkpoint["generator"]`
- some checkpoint tensor names / shapes are adapted automatically when loading the streaming model

## Quick start

All commands below are run from the repository root.

### 1. PyTorch streaming inference

Run the streaming PyTorch model on a wav file:

```bash
python -m streaming.demo_cuda_graph \
  --checkpoint_file cp_model_casual_whamr/g_00120000 \
  --input_wav wavs/file_000_snr-5.wav \
  --output_wav generated_files/enh_stream_demo.wav \
  --device cuda \
  --cuda_graph
```

For CPU inference, use `--device cpu` and optionally set `--threads`.

### 2. Verify offline vs streaming consistency

This script compares the offline model and the frame-wise streaming model:

```bash
python check_offline_vs_stream.py
```

It reports differences in:

- magnitude
- phase
- complex spectrum
- reconstructed waveform

This is useful after modifying streaming caches or causal blocks.

### 3. Export streaming model to ONNX

```bash
python export_stream_onnx.py \
  --checkpoint_file cp_model_casual_whamr/g_00120000 \
  --onnx_file cp_model_casual_whamr/stream.onnx \
  --opset 17
```

The exported ONNX model contains the streaming caches as explicit inputs/outputs.

If you want to convert a trained offline checkpoint to the streaming deployment format, run a command like:

```bash
python export_stream_onnx.py --checkpoint_file E:\a\codes\RENet-causal\cp_model_casual_whamr\g_00180000
```

This generates the streaming ONNX model used by the later TensorRT steps.

### 4. Run ONNX Runtime streaming inference

#### CUDA provider

```bash
python -m streaming.onnx_app \
  --onnx_file cp_model_casual_whamr/stream.onnx \
  --config_file cp_model_casual_whamr/config.json \
  --input_wav wavs/file_000_snr-5.wav \
  --provider cuda \
  --use_iobinding \
  --output_wav generated_files/enh_onnx_cuda.wav
```

#### TensorRT execution provider

```bash
python -m streaming.onnx_app \
  --onnx_file cp_model_casual_whamr/stream.onnx \
  --config_file cp_model_casual_whamr/config.json \
  --input_wav wavs/file_000_snr-5.wav \
  --provider trt \
  --use_iobinding \
  --strict_trt_ep \
  --trt_cache_dir generated_files/trt_cache \
  --output_wav generated_files/enh_onnx_trt.wav
```

Useful options:

- `--trt_fp16`: enable TensorRT FP16 engine build
- `--strict_trt_ep`: disallow CUDA / CPU fallback
- `--strict_cuda_ep`: disallow CPU fallback for CUDA EP
- `--ort_log_severity`: change ONNX Runtime log verbosity

The script reports total elapsed time and **RTF** (real-time factor).

### Recommended TensorRT build procedure

The typical TensorRT workflow in this repository is:

1. **Install TensorRT first**.
2. Export the trained checkpoint to streaming ONNX.
3. Run `streaming.onnx_app` with TensorRT EP once to trigger engine building / caching.
4. Use the generated TensorRT engine for realtime testing.

Example:

```bash
python export_stream_onnx.py --checkpoint_file E:\a\codes\RENet-causal\cp_model_casual_whamr\g_00180000
python -m streaming.onnx_app --input_wav wavs/file_000_snr-5.wav --use_iobinding --strict_trt_ep
```

After the first successful run, TensorRT engine files are typically cached under `generated_files/trt_cache/`.

### 5. Run a serialized TensorRT engine directly

If you already have a TensorRT engine file:

```bash
python -m streaming.trt_cuda_graph_app \
  --engine_file generated_files/trt_cache/TensorrtExecutionProvider_TRTKernel_graph_main_graph_1772623164867571760_0_0_sm89.engine \
  --config_file cp_model_casual_whamr/config.json \
  --input_wav wavs/file_000_snr-5.wav \
  --output_wav generated_files/enh_trt_cuda_graph.wav \
  --use_cuda_graph
```

This path bypasses ONNX Runtime and executes the TensorRT engine directly.

### 6. Run the real-time microphone demo

First list audio devices:

```bash
python -m streaming.realtime_trt_demo --list_devices
```

Then start the real-time demo:

```bash
python -m streaming.realtime_trt_demo \
  --use_cuda_graph \
  --input_device 4 \
  --output_device 20 \
  --queue_blocks 96 \
  --prefill_blocks 12 \
  --latency high \
  --device_sr 48000 \
  --device_blocksize 2048 \
  --enter_toggle_raw
```

Adjust `--input_device` and `--output_device` according to your own physical or virtual audio devices.

Useful real-time options:

- `--queue_blocks`: input/output queue capacity
- `--prefill_blocks`: playback prefill depth
- `--device_sr`: audio interface sample rate
- `--device_blocksize`: callback block size
- `--latency`: sounddevice latency mode
- `--enter_toggle_raw`: press Enter to toggle denoised/original output
- `--bypass_model`: diagnose the audio device path without running TensorRT

## Notes on TensorRT engines

TensorRT engine files are usually **machine-specific**.

They may depend on:

- GPU architecture
- CUDA version
- TensorRT version
- ONNX Runtime TensorRT EP version / builder settings

So cached `.engine` files under `generated_files/trt_cache/` should be treated as local build artifacts, not portable release assets.

## Expected outputs

Typical generated files include:

- enhanced wavs under `generated_files/` or `wavs/`
- ONNX Runtime profiles under `generated_files/ort_profile/`
- TensorRT engine caches under `generated_files/trt_cache/`

## Current scope and limitations

- This repository is mainly for **streaming inference and deployment experiments**.
- The full training, dataset preparation, and benchmark pipeline live in the original RENet repository.
- The model is still relatively heavy for very constrained edge devices; it is more suitable for GPU-equipped or cloud-side real-time deployment.
- The streaming implementation is practical and usable, but still a good base for further kernel fusion, cache optimization, and latency tuning.

## Relationship to the original RENet repository

If you need the complete research codebase for:

- training
- non-streaming inference
- evaluation scripts
- dataset preparation
- paper-level experiment reproduction

please use the upstream repository:

<https://github.com/wangchengzhong/RENet>

This repository should be viewed as the **deployment / streaming branch** built around that work.

## License

This repository is released under the MIT License.

See [LICENSE](LICENSE) for details.
