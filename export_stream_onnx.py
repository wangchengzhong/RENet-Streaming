from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os

import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

from models.model import StreamMPNet


def load_checkpoint(filepath, device):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)
    return torch.load(filepath, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--onnx_file", default="./cp_model_casual_whamr/stream.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic_batch", action="store_true", help="Allow dynamic batch axis; default is strict static B=1")
    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], "config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        h = AttrDict(json.loads(f.read()))

    device = torch.device("cpu")
    ckpt = load_checkpoint(args.checkpoint_file, device)

    stream_model = StreamMPNet(h).to(device).eval()
    # stream_model.mpnet.load_state_dict(ckpt["generator"])

    ckpt_state = ckpt["generator"]
    model_state = stream_model.mpnet.state_dict()

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

    load_result = stream_model.mpnet.load_state_dict(matched_state, strict=True)
    print(
        f"[Checkpoint] loaded {len(matched_state)} tensors, "
        f"skipped {skipped_missing_or_mismatch} by key/shape mismatch, "
        f"missing_in_ckpt={len(load_result.missing_keys)}, "
        f"unexpected_in_ckpt={len(load_result.unexpected_keys)}"
    )

    n_freqs = h.n_fft // 2 + 1
    noisy_amp_t = torch.randn(1, n_freqs, 1, device=device)
    noisy_pha_t = torch.randn(1, n_freqs, 1, device=device)
    caches = stream_model.init_stream_cache(1, n_freqs, device=device)

    os.makedirs(os.path.dirname(args.onnx_file) or ".", exist_ok=True)
    dynamic_axes = None

    torch.onnx.export(
        stream_model,
        (noisy_amp_t, noisy_pha_t, *caches),
        args.onnx_file,
        input_names=[
            "noisy_amp_t", "noisy_pha_t",
            "enc_cache_1", "enc_cache_2", "enc_cache_3", "enc_cache_4",
            "dec_cache_1", "dec_cache_2", "dec_cache_3", "dec_cache_4",
            "ts_k_cache",
            "ts_v_cache",
            "ts_kv_cache",
            "ts_attn_count",
            "ts_gru_cache",
            "ts_ang_in_cache_r",
            "ts_ang_in_cache_i",
            "ts_ang_mid_cache_r",
            "ts_ang_mid_cache_i",
            "enc_amp_power_cache",
            "enc_ang_power_cache",
            "enc_amp_count_cache",
            "enc_ang_count_cache",
            "dec_amp_power_cache",
            "dec_ang_power_cache",
            "dec_amp_count_cache",
            "dec_ang_count_cache",
            "input_amp_power_cache",
            "input_amp_count_cache",
        ],
        output_names=[
            "denoised_amp_t", "denoised_pha_t", "denoised_com_t",
            "enc_cache_1_out", "enc_cache_2_out", "enc_cache_3_out", "enc_cache_4_out",
            "dec_cache_1_out", "dec_cache_2_out", "dec_cache_3_out", "dec_cache_4_out",
            "ts_k_cache_out",
            "ts_v_cache_out",
            "ts_kv_cache_out",
            "ts_attn_count_out",
            "ts_gru_cache_out",
            "ts_ang_in_cache_r_out",
            "ts_ang_in_cache_i_out",
            "ts_ang_mid_cache_r_out",
            "ts_ang_mid_cache_i_out",
            "enc_amp_power_cache_out",
            "enc_ang_power_cache_out",
            "enc_amp_count_cache_out",
            "enc_ang_count_cache_out",
            "dec_amp_power_cache_out",
            "dec_ang_power_cache_out",
            "dec_amp_count_cache_out",
            "dec_ang_count_cache_out",
            "input_amp_power_cache_out",
            "input_amp_count_cache_out",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )

    mode = "dynamic-batch" if args.dynamic_batch else "strict-static-b1t1"
    print(f"ONNX export done ({mode}): {args.onnx_file}")


if __name__ == "__main__":
    main()
