# coding=utf-8
"""Video inference with ONNX vision encoder + llama.cpp ctypes decoder."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_config import DEFAULT_EXPORT_DIR, V45_LLM_GGUF, V46_LLM_GGUF
from minicpmv_llama import LlamaModel, LlamaContext, LlamaSampler, LlamaBatch
from preprocess import (
    preprocess_frame, extract_frames,
    get_2d_sincos_pos_embed_numpy, compute_temporal_embeddings_for_group,
    encode_video_temporal_ids, group_temporal_ids,
)

PATCH_SIZE = 14
V45_NPPS = 70
V46_NPPS = 70
V45_EMBED_DIM = 4096


def provider_list(provider):
    if provider == "cuda":
        return ["CUDAExecutionProvider"]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def check_provider(sess, name, provider):
    actual = sess.get_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in actual:
        raise RuntimeError(f"{name} ONNX did not bind CUDA provider: {actual}")
    return actual


def output_name(sess):
    return sess.get_outputs()[0].name


def get_video_fps(video_path):
    import av
    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)
    container.close()
    return fps


def load_ort_session(path, provider):
    import onnxruntime as ort
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=provider_list(provider))


def preprocess_frames(frames, version, max_slice_nums=9):
    tiles = []
    for frame_idx, frame in enumerate(frames):
        frame_tiles = preprocess_frame(
            frame,
            frame_idx=frame_idx,
            version=version,
            max_slice_nums=max_slice_nums,
        )
        tiles.extend(frame_tiles)
    return tiles


def encode_v45_video(frames, export_dir, fps, provider, packing_size=1, resampler_dtype="fp16"):
    from importlib import import_module
    export_mod = import_module("01-Export-Vision-Encoder")

    siglip_path = Path(export_dir) / "minicpmv_v45_siglip.fp32.onnx"
    resampler_path = Path(export_dir) / f"minicpmv_v45_resampler_temporal.{resampler_dtype}.onnx"
    siglip_sess = load_ort_session(siglip_path, provider)
    resampler_sess = load_ort_session(resampler_path, provider)
    siglip_output = output_name(siglip_sess)
    resampler_output = output_name(resampler_sess)
    resampler_np_dtype = np.float16 if "float16" in resampler_sess.get_inputs()[0].type else np.float32
    print(
        f"  ONNX providers: SigLIP={check_provider(siglip_sess, 'SigLIP', provider)}, "
        f"Resampler={check_provider(resampler_sess, 'Resampler', provider)}"
    )
    print(f"  V4.5 ONNX: SigLIP=fp32, temporal Resampler={resampler_dtype}")

    tiles = preprocess_frames(frames, version="4.5", max_slice_nums=1)
    siglip_features = []
    patch_counts = []
    for tile in tiles:
        h, w = tile["h"], tile["w"]
        inputs = export_mod.compute_onnx_inputs_v45(h, w, V45_NPPS)
        feat = siglip_sess.run([siglip_output], {
            "pixel_values": tile["pixel_values"].astype(np.float32),
            "pos_ids": inputs["pos_ids"],
        })[0]
        siglip_features.append(feat)
        patch_counts.append(h * w)

    frame_indices = np.linspace(0, len(frames) - 1, len(frames), dtype=int)
    temporal_ids = encode_video_temporal_ids(frame_indices, fps)
    temporal_groups = group_temporal_ids(temporal_ids, packing_size)

    all_tokens = []
    offset = 0
    for group_tids in temporal_groups:
        group_size = len(group_tids)
        group_feats = np.concatenate(siglip_features[offset:offset + group_size], axis=0)

        spatial_parts = []
        for tile in tiles[offset:offset + group_size]:
            h, w = tile["h"], tile["w"]
            spe = get_2d_sincos_pos_embed_numpy(V45_EMBED_DIM, (h, w)).reshape(h * w, -1)
            spatial_parts.append(spe)
        spatial_embeds = np.concatenate(spatial_parts, axis=0)

        temporal_embeds = compute_temporal_embeddings_for_group(
            patch_counts[offset:offset + group_size],
            group_tids,
            V45_EMBED_DIM,
        )
        visual_tokens = resampler_sess.run([resampler_output], {
            "siglip_features": group_feats.astype(resampler_np_dtype),
            "spatial_pos_embeds": spatial_embeds.astype(resampler_np_dtype),
            "temporal_pos_embeds": temporal_embeds.astype(resampler_np_dtype),
        })[0]
        all_tokens.append(visual_tokens.astype(np.float32))
        offset += group_size

    return np.concatenate(all_tokens, axis=0), len(tiles), len(all_tokens)


def encode_v45_per_tile(frames, export_dir, provider, resampler_dtype="fp32"):
    from importlib import import_module
    export_mod = import_module("01-Export-Vision-Encoder")

    siglip_path = Path(export_dir) / "minicpmv_v45_siglip.fp32.onnx"
    resampler_path = Path(export_dir) / f"minicpmv_v45_resampler.{resampler_dtype}.onnx"
    siglip_sess = load_ort_session(siglip_path, provider)
    resampler_sess = load_ort_session(resampler_path, provider)
    siglip_output = output_name(siglip_sess)
    resampler_output = output_name(resampler_sess)
    resampler_np_dtype = np.float16 if "float16" in resampler_sess.get_inputs()[0].type else np.float32
    print(
        f"  ONNX providers: SigLIP={check_provider(siglip_sess, 'SigLIP', provider)}, "
        f"Resampler={check_provider(resampler_sess, 'Resampler', provider)}"
    )
    print(f"  V4.5 ONNX: SigLIP=fp32, per-tile Resampler={resampler_dtype}")

    tiles = preprocess_frames(frames, version="4.5", max_slice_nums=9)
    all_tokens = []
    for tile in tiles:
        h, w = tile["h"], tile["w"]
        inputs = export_mod.compute_onnx_inputs_v45(h, w, V45_NPPS)
        feat = siglip_sess.run([siglip_output], {
            "pixel_values": tile["pixel_values"].astype(np.float32),
            "pos_ids": inputs["pos_ids"],
        })[0]

        spatial_embeds = get_2d_sincos_pos_embed_numpy(V45_EMBED_DIM, (h, w)).reshape(h * w, -1)
        visual_tokens = resampler_sess.run([resampler_output], {
            "siglip_features": feat.astype(resampler_np_dtype),
            "spatial_pos_embeds": spatial_embeds.astype(resampler_np_dtype),
        })[0]
        all_tokens.append(visual_tokens.astype(np.float32))

    return np.concatenate(all_tokens, axis=0), len(tiles), len(tiles)


def encode_v46_video(frames, export_dir, provider):
    from importlib import import_module
    export_mod = import_module("01-Export-Vision-Encoder")

    onnx_path = Path(export_dir) / "minicpmv_vision_encoder.fp16.onnx"
    sess = load_ort_session(onnx_path, provider)
    vision_output = output_name(sess)
    print(f"  ONNX providers: Vision={check_provider(sess, 'Vision', provider)}")

    tiles = preprocess_frames(frames, version="4.6")
    all_tokens = []
    for tile in tiles:
        h, w = tile["h"], tile["w"]
        indices = export_mod.compute_onnx_indices(h, w, V46_NPPS)
        feed = {
            "pixel_values": tile["pixel_values"].astype(np.float16),
            "pos_ids": indices["pos_ids"],
            "window_index": indices["window_index"],
            "window_sort_idx": indices["window_sort_idx"],
            "merge_index": indices["merge_index"],
            "ds_index": indices["ds_index"],
        }
        tokens = sess.run([vision_output], feed)[0]
        all_tokens.append(tokens.astype(np.float32))
    return np.concatenate(all_tokens, axis=0), len(tiles), None


def generate(model, ctx, sampler, prompt_tokens, visual_tokens, n_predict):
    pad_id = model.tokenize("<|image_pad|>", parse_special=True)[0]
    n_visual = visual_tokens.shape[0]

    segments = []
    text = []
    for tok in prompt_tokens:
        if tok == pad_id:
            if text:
                segments.append(("text", text))
                text = []
            if segments and segments[-1][0] == "visual":
                segments[-1] = ("visual", segments[-1][1] + 1)
            else:
                segments.append(("visual", 1))
        else:
            text.append(tok)
    if text:
        segments.append(("text", text))

    total_vis = sum(value for seg_type, value in segments if seg_type == "visual")
    if total_vis != n_visual:
        raise RuntimeError(f"visual token mismatch: prompt has {total_vis} pads, embeddings have {n_visual}")

    pos = 0
    vis_idx = 0
    for seg_type, value in segments:
        if seg_type == "text":
            batch = LlamaBatch(len(value), embd_dim=0)
            batch.set_tokens(value, pos_offset=pos)
            ctx.decode(batch)
            pos += len(value)
        else:
            batch = LlamaBatch(value, embd_dim=model.n_embd)
            batch.set_embd(visual_tokens[vis_idx:vis_idx + value], pos_offset=pos)
            ctx.decode(batch)
            vis_idx += value
            pos += value

    raw = bytearray()
    for _ in range(n_predict):
        tok = sampler.sample(ctx)
        sampler.accept(tok)
        if tok == model.eos_token:
            break
        raw.extend(model.detokenize(tok))
        if ctx.decode_token(tok, pos=pos) != 0:
            break
        pos += 1
    return raw.decode("utf-8", errors="replace")


def build_v45_prompt(visual_tokens, question):
    n_units = visual_tokens.shape[0] // 64
    unit = "<image>" + "<|image_pad|>" * 64 + "</image>"
    return f"<|im_start|>user\n{unit * n_units}\n{question}<|im_end|>\n<|im_start|>assistant\n"


def build_v46_prompt(visual_tokens, question):
    return (
        f"<|im_start|>user\n"
        f"{'<|image_pad|>' * visual_tokens.shape[0]}\n"
        f"{question}<|im_end|><|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="sample_video.mp4")
    parser.add_argument("--version", choices=["4.5", "4.6"], default="4.5")
    parser.add_argument("--gguf", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--llama-gpu-layers", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--num-frames", type=int, default=7)
    parser.add_argument("--packing", type=int, default=1)
    parser.add_argument("--resampler-dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--no-temporal", action="store_true", help="Use legacy per-tile V4.5 Resampler")
    parser.add_argument("--question", type=str, default="请详细描述这个视频的内容。")
    parser.add_argument("--n-predict", type=int, default=200)
    args = parser.parse_args()

    gguf = args.gguf or str(V45_LLM_GGUF if args.version == "4.5" else V46_LLM_GGUF)
    print(f"Mode: MiniCPM-V {args.version}")
    print(f"Video: {args.video}")

    t0 = time.time()
    fps = get_video_fps(args.video)
    frames = extract_frames(args.video, num_frames=args.num_frames)
    if args.version == "4.5":
        if args.no_temporal:
            visual_tokens, tile_count, package_count = encode_v45_per_tile(
                frames,
                args.export_dir,
                args.provider,
                resampler_dtype=args.resampler_dtype,
            )
        else:
            visual_tokens, tile_count, package_count = encode_v45_video(
                frames,
                args.export_dir,
                fps,
                args.provider,
                packing_size=args.packing,
                resampler_dtype=args.resampler_dtype,
            )
        print(
            f"  {len(frames)} frames, {tile_count} tiles, packing={args.packing} -> "
            f"{package_count} packages x 64 = {visual_tokens.shape[0]} visual tokens ({time.time() - t0:.1f}s)"
        )
        prompt = build_v45_prompt(visual_tokens, args.question)
    else:
        visual_tokens, tile_count, _ = encode_v46_video(frames, args.export_dir, args.provider)
        print(
            f"  {len(frames)} frames, {tile_count} tiles -> "
            f"{visual_tokens.shape[0]} visual tokens ({time.time() - t0:.1f}s)"
        )
        prompt = build_v46_prompt(visual_tokens, args.question)

    n_vis = visual_tokens.shape[0]
    n_batch = max(4096, n_vis + 256)
    n_ctx = max(args.ctx_size, n_vis + 1024)

    print(f"Loading GGUF: {gguf}")
    model = LlamaModel(gguf, n_gpu_layers=args.llama_gpu_layers)
    ctx = LlamaContext(model, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_batch)
    sampler = LlamaSampler(temperature=0, repeat_penalty=1.1)
    print(f"  n_embd={model.n_embd}, n_vocab={model.n_vocab}, n_ctx={n_ctx}, n_batch={n_batch}")

    prompt_tokens = model.tokenize(prompt, add_special=True, parse_special=True)
    answer = generate(model, ctx, sampler, prompt_tokens, visual_tokens, args.n_predict)

    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    ctx.clear_kv()
    sampler.free()


if __name__ == "__main__":
    main()
