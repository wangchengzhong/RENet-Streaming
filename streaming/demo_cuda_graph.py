import argparse
import time
from pathlib import Path

from streaming.engine import MPNetStreamingEnhancer


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated streaming MPNet demo")
    parser.add_argument("--checkpoint_file", default="cp_model_casual_whamr/g_00180000")
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="generated_files/enh_stream_demo.wav")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA Graph replay for per-frame streaming step")
    args = parser.parse_args()

    t0 = time.perf_counter()
    enhancer = MPNetStreamingEnhancer(
        checkpoint_file=args.checkpoint_file,
        config_file=args.config_file,
        device=args.device,
        num_threads=args.threads,
        use_cuda_graph=args.cuda_graph,
    )
    t1 = time.perf_counter()

    stat = enhancer.enhance_file(args.input_wav, args.output_wav)
    t2 = time.perf_counter()

    print(f"Loaded model in {(t1 - t0):.2f}s")
    print(f"Enhanced file in {(t2 - t1):.2f}s")
    print(f"Streaming elapsed: {stat['elapsed_sec']:.3f}s, audio: {stat['audio_sec']:.3f}s, RTF: {stat['rtf']:.4f}")
    print(f"Saved: {Path(args.output_wav).resolve()}")


if __name__ == "__main__":
    main()
