# coding=utf-8
"""Export MiniCPM-V vision-side FP32 ONNX files.

V4.5 exports the reproducible baseline graphs:
  - minicpmv_v45_siglip.fp32.onnx
  - minicpmv_v45_resampler.fp32.onnx
  - minicpmv_v45_resampler_temporal.fp32.onnx

V4.5 FP16 Resampler variants are derived by 03-Quantize-Vision-Encoder.py.
V4.6 exports the single-file dynamic vision encoder.
"""
import os
import sys
import types
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_config import DEFAULT_EXPORT_DIR, V45_MODEL_DIR, V46_MODEL_DIR

PATCH_SIZE = 14


def _file_size_mb(path):
    path = Path(path)
    total = path.stat().st_size if path.exists() else 0
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / (1024 * 1024)


def _eager_attn(q_proj, k_proj, v_proj, hidden_states, num_heads, head_dim):
    q = q_proj(hidden_states).view(1, -1, num_heads, head_dim).transpose(1, 2)
    k = k_proj(hidden_states).view(1, -1, num_heads, head_dim).transpose(1, 2)
    v = v_proj(hidden_states).view(1, -1, num_heads, head_dim).transpose(1, 2)
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(2, 3)) * scale
    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v).transpose(1, 2).contiguous().view(1, -1, num_heads * head_dim)


def _export_onnx(wrapper, dummy_inputs, onnx_path, input_names, output_names, dynamic_axes, dynamo=False):
    print(f"Exporting ONNX to {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=False,
            dynamo=dynamo,
        )
    print(f"Done: {onnx_path} ({_file_size_mb(onnx_path):.1f} MB)")


def get_2d_sincos_pos_embed_numpy(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid_numpy(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_numpy(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid_new_numpy(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new_numpy(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid_new_numpy(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    out = np.einsum("hw,d->hwd", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


def compute_onnx_inputs_v45(h, w, npps=70, resampler_embed_dim=4096):
    """Compute ONNX inputs for V4.5 from patch grid (h, w)."""
    bucket_h = np.clip((np.arange(h) * npps) // h, 0, npps - 1)
    bucket_w = np.clip((np.arange(w) * npps) // w, 0, npps - 1)
    pos_ids = (bucket_h[:, None] * npps + bucket_w).flatten().astype(np.int64)

    spatial_pos_embed = get_2d_sincos_pos_embed_numpy(resampler_embed_dim, (h, w))
    spatial_pos_embed = spatial_pos_embed.reshape(h * w, -1).astype(np.float32)

    return {
        "pos_ids": pos_ids,
        "spatial_pos_embed": spatial_pos_embed,
    }


def load_local_minicpmv_class(model_dir):
    """Load V4.5 MiniCPMV from local model code without modifying models/."""
    model_dir = Path(model_dir).resolve()
    if not (model_dir / "modeling_minicpmv.py").exists():
        nested = model_dir / "MiniCPM-V-4_5"
        if (nested / "modeling_minicpmv.py").exists():
            model_dir = nested
    package_name = "_minicpmv_v45_local"
    module_name = f"{package_name}.modeling_minicpmv"

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(model_dir)]
        sys.modules[package_name] = pkg

    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, model_dir / "modeling_minicpmv.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    cls = sys.modules[module_name].MiniCPMV
    # transformers 5.x expects all_tied_weights_keys; local V4.5 model only has _tied_weights_keys
    cls.all_tied_weights_keys = property(lambda self: {})
    return cls


def load_v45_model_for_export(model_dir):
    """Load V4.5 and explicitly restore visual-side weights from safetensors.

    The local V4.5 `from_pretrained` path can leave `vpm.*` randomly initialized
    on recent Transformers builds.  For reproducible ONNX export, always overlay
    the visual encoder and resampler weights directly from the checkpoint shards.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir).resolve()
    if not (model_dir / "modeling_minicpmv.py").exists():
        nested = model_dir / "MiniCPM-V-4_5"
        if (nested / "modeling_minicpmv.py").exists():
            model_dir = nested
    MiniCPMV = load_local_minicpmv_class(model_dir)
    model = MiniCPMV.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()

    shard_paths = sorted(model_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors shards found in {model_dir}")

    prefixes = ("vpm.", "resampler.")
    state = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefixes):
                    state[key] = f.get_tensor(key).to(torch.float32)

    expected = [k for k in model.state_dict().keys() if k.startswith(prefixes)]
    missing = sorted(k for k in expected if k not in state)
    if missing:
        raise RuntimeError(f"Missing visual checkpoint weights: {missing[:8]} ... total={len(missing)}")

    incompatible = model.load_state_dict(state, strict=False)
    bad_unexpected = [k for k in incompatible.unexpected_keys if k.startswith(prefixes)]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected visual checkpoint weights: {bad_unexpected[:8]}")

    patch_std = float(model.vpm.embeddings.patch_embedding.weight.detach().float().std())
    print(f"  loaded visual weights: {len(state)} tensors, patch_embedding.std={patch_std:.6f}")
    return model.float().eval()


def compute_onnx_indices(h, w, npps):
    """Compute all indices needed for ONNX inference from patch grid (h, w).

    Args:
        h: patch grid height (must be divisible by 4)
        w: patch grid width (must be divisible by 4)
        npps: num_patches_per_side from NaViT config (10 for MiniCPM-V 4.6)

    Returns:
        dict with numpy arrays: pos_ids, window_index, window_sort_idx,
        merge_index, ds_index
    """
    num_patches = h * w

    # --- pos_ids (NaViT position embedding via bucketize) ---
    # Equivalent to torch.bucketize but in pure numpy:
    #   bucketize(i/h, boundaries, right=True) = floor(i * npps / h)
    bucket_h = np.clip((np.arange(h) * npps) // h, 0, npps - 1)
    bucket_w = np.clip((np.arange(w) * npps) // w, 0, npps - 1)
    pos_ids = (bucket_h[:, None] * npps + bucket_w).flatten().astype(np.int64)

    # --- window_index (vit_merger window attention, 2×2 windows) ---
    # Groups patches into 2×2 windows for local attention
    index = np.arange(num_patches).reshape(h, w)
    nwh, nww = h // 2, w // 2
    index = index.reshape(nwh, 2, nww, 2).transpose(0, 2, 1, 3).reshape(nwh * nww, 4)
    window_index = index.flatten().astype(np.int64)
    window_sort_idx = np.argsort(window_index).astype(np.int64)

    # --- merge_index (vit_merger 2×2 spatial concat after window attention) ---
    # Same grouping as window_index but used for spatial concatenation
    grid = np.arange(num_patches).reshape(h, w)
    mh, mw = h // 2, w // 2
    merge_index = grid.reshape(mh, 2, mw, 2).transpose(0, 2, 1, 3).reshape(-1, 4).astype(np.int64)

    # --- ds_index (downsample 2×2 spatial concat) ---
    # Applied to the vit_merger output grid (mh × mw)
    ds_grid = np.arange(mh * mw).reshape(mh, mw)
    ds_mh, ds_mw = mh // 2, mw // 2
    ds_index = ds_grid.reshape(ds_mh, 2, ds_mw, 2).transpose(0, 2, 1, 3).reshape(-1, 4).astype(np.int64)

    return {
        "pos_ids": pos_ids,
        "window_index": window_index,
        "window_sort_idx": window_sort_idx,
        "merge_index": merge_index,
        "ds_index": ds_index,
    }


class VisionEncoderONNX(nn.Module):
    """NaViT → SigLIP (27 layers + vit_merger at layer 6) → DownsampleMLP

    Dynamic version: all spatial indices are passed as forward() inputs.
    No precomputed buffers — works for any valid (h, w).
    """

    def __init__(self, vision_tower, merger):
        super().__init__()
        self.embeddings = vision_tower.embeddings
        self.encoder = vision_tower.encoder
        self.post_layernorm = vision_tower.post_layernorm
        self.vit_merger = vision_tower.vit_merger
        self.merger = merger
        self.insert_layer_id = vision_tower.config.insert_layer_id
        self.hidden_size = vision_tower.config.hidden_size

    def forward(self, pixel_values, pos_ids, window_index, window_sort_idx, merge_index, ds_index):
        # Patch embedding
        patch_embeds = self.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.embeddings.position_embedding(pos_ids).unsqueeze(0)

        # SigLIP encoder layers
        hidden_states = embeddings
        for layer_idx, layer in enumerate(self.encoder.layers):
            residual = hidden_states
            hidden_states = layer.layer_norm1(hidden_states)
            attn_out = _eager_attn(
                layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                hidden_states, layer.self_attn.num_heads, layer.self_attn.head_dim,
            )
            hidden_states = residual + layer.self_attn.out_proj(attn_out)

            residual = hidden_states
            hidden_states = layer.layer_norm2(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

            if layer_idx == self.insert_layer_id:
                hidden_states = self._vit_merger(hidden_states, window_index, window_sort_idx, merge_index)

        hidden_states = self.post_layernorm(hidden_states)
        return self._downsample_mlp(hidden_states, ds_index)

    def _vit_merger(self, hidden_states, window_index, window_sort_idx, merge_index):
        residual = hidden_states
        hidden_states = self.vit_merger.layer_norm1(hidden_states)

        # Window attention: permute → attn → unsort
        windowed = hidden_states[:, window_index, :]
        attn_out = _eager_attn(
            self.vit_merger.self_attn.q_proj, self.vit_merger.self_attn.k_proj,
            self.vit_merger.self_attn.v_proj, windowed,
            self.vit_merger.self_attn.num_heads, self.vit_merger.self_attn.head_dim,
        )
        attn_out = self.vit_merger.self_attn.out_proj(attn_out)[:, window_sort_idx, :]
        hidden_states = residual + attn_out

        # 2×2 spatial merge using precomputed index (replaces view+permute)
        p = hidden_states.squeeze(0)                              # [num_patches, dim]
        windowed = p[merge_index]                                 # [num_windows, 4, dim]
        cat = windowed.reshape(-1, 4 * self.hidden_size)          # [num_windows, 4*dim]
        res = windowed.mean(dim=1)                                # [num_windows, dim]
        cat = self.vit_merger.act(self.vit_merger.linear_1(self.vit_merger.pre_norm(cat)))
        return (self.vit_merger.linear_2(cat) + res).unsqueeze(0)

    def _downsample_mlp(self, hidden_states, ds_index):
        mlp = self.merger.mlp[0]
        p = hidden_states.squeeze(0)                              # [num_patches, dim]
        windowed = p[ds_index]                                    # [num_windows, 4, dim]
        cat = windowed.reshape(-1, 4 * p.shape[-1])               # [num_windows, 4*dim]
        return mlp(cat)


class SigLIPEasyV45ONNX(nn.Module):
    """V4.5 SigLIP export wrapper with manual LayerNorm and position lookup.

    This avoids TorchScript ONNX export bugs observed with `nn.LayerNorm` and
    `nn.Embedding` on the MiniCPM-V 4.5 vision stack.
    """

    def __init__(self, vision_tower):
        super().__init__()
        patch = vision_tower.embeddings.patch_embedding
        self.register_buffer("patch_emb_w", patch.weight.detach().float().clone())
        self.register_buffer("patch_emb_b", patch.bias.detach().float().clone())
        self.register_buffer("pos_emb_w", vision_tower.embeddings.position_embedding.weight.detach().float().clone())
        self.register_buffer("post_ln_w", vision_tower.post_layernorm.weight.detach().float().clone())
        self.register_buffer("post_ln_b", vision_tower.post_layernorm.bias.detach().float().clone())

        self.patch_stride = patch.stride if isinstance(patch.stride, str) else tuple(patch.stride)
        self.patch_padding = patch.padding if isinstance(patch.padding, str) else tuple(patch.padding)
        self.patch_dilation = patch.dilation if isinstance(patch.dilation, str) else tuple(patch.dilation)
        self.patch_groups = int(patch.groups)
        self.encoder = vision_tower.encoder
        self.post_ln_eps = float(vision_tower.post_layernorm.eps)

    @staticmethod
    def _manual_layernorm(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(var + eps) * weight + bias

    def forward(self, pixel_values, pos_ids):
        patch_embeds = F.conv2d(
            pixel_values,
            self.patch_emb_w,
            self.patch_emb_b,
            self.patch_stride,
            self.patch_padding,
            self.patch_dilation,
            self.patch_groups,
        )
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.pos_emb_w[pos_ids].unsqueeze(0)

        hidden_states = embeddings
        for layer in self.encoder.layers:
            residual = hidden_states
            hidden_states = self._manual_layernorm(
                hidden_states,
                layer.layer_norm1.weight,
                layer.layer_norm1.bias,
                float(layer.layer_norm1.eps),
            )
            attn_out = _eager_attn(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                hidden_states,
                layer.self_attn.num_heads,
                layer.self_attn.head_dim,
            )
            hidden_states = residual + layer.self_attn.out_proj(attn_out)

            residual = hidden_states
            hidden_states = self._manual_layernorm(
                hidden_states,
                layer.layer_norm2.weight,
                layer.layer_norm2.bias,
                float(layer.layer_norm2.eps),
            )
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self._manual_layernorm(hidden_states, self.post_ln_w, self.post_ln_b, self.post_ln_eps)
        return hidden_states.squeeze(0)


class ResamplerV45ONNX(nn.Module):
    """V4.5 Resampler only. Batched per-frame (all tiles concatenated)."""

    def __init__(self, resampler):
        super().__init__()
        self.resampler = resampler

    def _forward_impl(self, siglip_features, spatial_pos_embeds, temporal_pos_embeds=None):
        x = self.resampler.kv_proj(siglip_features.unsqueeze(0))
        x = self.resampler.ln_kv(x).squeeze(0)

        q = self.resampler.ln_q(self.resampler.query)
        k = x + spatial_pos_embeds
        if temporal_pos_embeds is not None:
            k = k + temporal_pos_embeds
        v = x

        embed_dim = self.resampler.embed_dim
        num_heads = self.resampler.num_heads
        head_dim = embed_dim // num_heads

        attn = self.resampler.attn
        w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)

        q_proj = (q @ w_q.T + b_q).reshape(1, 64, num_heads, head_dim).permute(0, 2, 1, 3)
        k_proj = (k @ w_k.T + b_k).reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v_proj = (v @ w_v.T + b_v).reshape(1, -1, num_heads, head_dim).permute(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q_proj, k_proj.transpose(2, 3)) * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v_proj)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().reshape(64, embed_dim)

        out = attn_out @ attn.out_proj.weight.T + attn.out_proj.bias
        out = self.resampler.ln_post(out.unsqueeze(0))
        out = (out @ self.resampler.proj).squeeze(0)
        return out

    def forward(self, siglip_features, spatial_pos_embeds):
        return self._forward_impl(siglip_features, spatial_pos_embeds)


class ResamplerTemporalV45ONNX(ResamplerV45ONNX):
    """V4.5 temporal Resampler. One ONNX call per clip/group."""

    def forward(self, siglip_features, spatial_pos_embeds, temporal_pos_embeds):
        return self._forward_impl(siglip_features, spatial_pos_embeds, temporal_pos_embeds)


ONNX_INPUT_NAMES = ["pixel_values", "pos_ids", "window_index", "window_sort_idx", "merge_index", "ds_index"]
ONNX_DYNAMIC_AXES = {
    "pixel_values":    {3: "total_patch_pixels"},
    "pos_ids":         {0: "num_patches"},
    "window_index":    {0: "num_patches"},
    "window_sort_idx": {0: "num_patches"},
    "merge_index":     {0: "num_windows"},
    "ds_index":        {0: "num_ds_windows"},
    "visual_tokens":   {0: "num_tokens"},
}

SIGLIP_V45_INPUT_NAMES = ["pixel_values", "pos_ids"]
SIGLIP_V45_DYNAMIC_AXES = {
    "pixel_values": {3: "total_patch_pixels"},
    "pos_ids": {0: "num_patches"},
    "siglip_features": {0: "num_patches"},
}

RESAMPLER_V45_INPUT_NAMES = ["siglip_features", "spatial_pos_embeds"]
RESAMPLER_V45_DYNAMIC_AXES = {
    "siglip_features": {0: "total_patches"},
    "spatial_pos_embeds": {0: "total_patches"},
    "visual_tokens": {},
}

RESAMPLER_TEMPORAL_V45_INPUT_NAMES = ["siglip_features", "spatial_pos_embeds", "temporal_pos_embeds"]
RESAMPLER_TEMPORAL_V45_DYNAMIC_AXES = {
    "siglip_features": {0: "total_patches"},
    "spatial_pos_embeds": {0: "total_patches"},
    "temporal_pos_embeds": {0: "total_patches"},
    "visual_tokens": {},
}


def make_dummy_inputs(h, w, npps, dtype=torch.float32):
    """Create dummy inputs for ONNX export tracing."""
    indices = compute_onnx_indices(h, w, npps)
    total_pixels = h * w * PATCH_SIZE
    return (
        torch.randn(1, 3, PATCH_SIZE, total_pixels, dtype=dtype),
        torch.tensor(indices["pos_ids"]),
        torch.tensor(indices["window_index"]),
        torch.tensor(indices["window_sort_idx"]),
        torch.tensor(indices["merge_index"]),
        torch.tensor(indices["ds_index"]),
    )


def make_dummy_inputs_siglip_v45(h, w, npps=32, dtype=torch.float32):
    inputs = compute_onnx_inputs_v45(h, w, npps)
    total_pixels = h * w * PATCH_SIZE
    return (
        torch.randn(1, 3, PATCH_SIZE, total_pixels, dtype=dtype),
        torch.tensor(inputs["pos_ids"]),
    )


def make_dummy_inputs_resampler_v45(total_patches=1024, dtype=torch.float32):
    return (
        torch.randn(total_patches, 1152, dtype=dtype),
        torch.randn(total_patches, 4096, dtype=dtype),
    )


def make_dummy_inputs_resampler_temporal_v45(total_patches=1024, dtype=torch.float32):
    return (
        torch.randn(total_patches, 1152, dtype=dtype),
        torch.randn(total_patches, 4096, dtype=dtype),
        torch.randn(total_patches, 4096, dtype=dtype),
    )


def export_vision_encoder(model_dir, export_dir, example_h=32, example_w=32):
    from transformers import MiniCPMV4_6ForConditionalGeneration

    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "minicpmv_vision_encoder.fp32.onnx")

    print(f"Loading model from {model_dir}...")
    model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()

    print(f"Building dynamic ONNX wrapper (tracing example: {example_h}x{example_w})...")
    wrapper = VisionEncoderONNX(model.model.vision_tower, model.model.merger).eval()

    npps = model.model.vision_tower.embeddings.num_patches_per_side
    dummy_inputs = make_dummy_inputs(example_h, example_w, npps)

    print(f"  pixel_values: {list(dummy_inputs[0].shape)}")
    with torch.no_grad():
        out = wrapper(*dummy_inputs)
    print(f"  output: {list(out.shape)}")

    num_tokens = (example_h // 4) * (example_w // 4)
    assert out.shape == (num_tokens, 1024), f"Shape mismatch: {out.shape} vs ({num_tokens}, 1024)"

    _export_onnx(
        wrapper,
        dummy_inputs,
        onnx_path,
        ONNX_INPUT_NAMES,
        ["visual_tokens"],
        ONNX_DYNAMIC_AXES,
    )


def export_siglip_v45(model_dir, export_dir, example_h=32, example_w=32, model=None):
    """Export V4.5 SigLIP ONNX (per-tile, FP32, manually normalized)."""
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "minicpmv_v45_siglip.fp32.onnx")

    if model is None:
        print(f"Loading V4.5 model from {model_dir}...")
        model = load_v45_model_for_export(model_dir)

    print(f"Building V4.5 SigLIPEasy ONNX wrapper (tracing example: {example_h}x{example_w})...")
    wrapper = SigLIPEasyV45ONNX(model.vpm).eval()

    npps = model.vpm.embeddings.num_patches_per_side
    dummy_inputs = make_dummy_inputs_siglip_v45(example_h, example_w, npps)

    print(f"  pixel_values: {list(dummy_inputs[0].shape)}")
    with torch.no_grad():
        out = wrapper(*dummy_inputs)
    print(f"  output: {list(out.shape)}")

    num_patches = example_h * example_w
    assert out.shape == (num_patches, 1152), f"Shape mismatch: {out.shape} vs ({num_patches}, 1152)"

    _export_onnx(
        wrapper,
        dummy_inputs,
        onnx_path,
        SIGLIP_V45_INPUT_NAMES,
        ["siglip_features"],
        SIGLIP_V45_DYNAMIC_AXES,
    )


def _export_resampler_v45(model_dir, export_dir, model=None, temporal=False):
    """Export V4.5 FP32 Resampler ONNX."""
    os.makedirs(export_dir, exist_ok=True)
    name = "minicpmv_v45_resampler_temporal.fp32.onnx" if temporal else "minicpmv_v45_resampler.fp32.onnx"
    onnx_path = os.path.join(export_dir, name)

    if model is None:
        print(f"Loading V4.5 model from {model_dir}...")
        model = load_v45_model_for_export(model_dir)

    label = "temporal Resampler" if temporal else "Resampler"
    print(f"Building V4.5 {label} ONNX wrapper...")
    wrapper_cls = ResamplerTemporalV45ONNX if temporal else ResamplerV45ONNX
    dummy_fn = make_dummy_inputs_resampler_temporal_v45 if temporal else make_dummy_inputs_resampler_v45
    input_names = RESAMPLER_TEMPORAL_V45_INPUT_NAMES if temporal else RESAMPLER_V45_INPUT_NAMES
    dynamic_axes = RESAMPLER_TEMPORAL_V45_DYNAMIC_AXES if temporal else RESAMPLER_V45_DYNAMIC_AXES

    wrapper = wrapper_cls(model.resampler).eval()
    dummy_inputs = dummy_fn()

    with torch.no_grad():
        out = wrapper(*dummy_inputs)
    print(f"  output: {list(out.shape)}")

    assert out.shape == (64, 4096), f"Shape mismatch: {out.shape} vs (64, 4096)"

    _export_onnx(wrapper, dummy_inputs, onnx_path, input_names, ["visual_tokens"], dynamic_axes)


def export_resampler_v45(model_dir, export_dir, model=None):
    return _export_resampler_v45(model_dir, export_dir, model=model, temporal=False)


def export_resampler_temporal_v45(model_dir, export_dir, model=None):
    return _export_resampler_v45(model_dir, export_dir, model=model, temporal=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["4.5", "4.6"], default="4.6")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--example-h", type=int, default=32, help="Example h for tracing")
    parser.add_argument("--example-w", type=int, default=32, help="Example w for tracing")
    args = parser.parse_args()

    model_dir = args.model_dir or str(V45_MODEL_DIR if args.version == "4.5" else V46_MODEL_DIR)
    if args.version == "4.5":
        print(f"Loading V4.5 model from {model_dir}...")
        model = load_v45_model_for_export(model_dir)
        export_siglip_v45(model_dir, args.export_dir, args.example_h, args.example_w, model=model)
        export_resampler_v45(model_dir, args.export_dir, model=model)
        export_resampler_temporal_v45(model_dir, args.export_dir, model=model)
    else:
        export_vision_encoder(model_dir, args.export_dir, args.example_h, args.example_w)
