# coding=utf-8
"""Export FP16 V4.5 Resampler ONNX files.

MiniCPM-V 4.5 uses a split vision path:

  SigLIP FP32 -> Resampler -> llama.cpp GGUF

Do not quantize or FP16-export the V4.5 SigLIP graph for runtime use.  The
27-layer SigLIP path is numerically sensitive and stays FP32.  The lightweight
Resampler and temporal 3D-Resampler can be exported as FP16 variants:

  - minicpmv_v45_resampler.fp16.onnx
  - minicpmv_v45_resampler_temporal.fp16.onnx

04-Verify-Video.py uses the FP16 temporal Resampler by default.  Use the FP32
files only when you intentionally want to compare the baseline export.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from export_config import DEFAULT_EXPORT_DIR, V45_MODEL_DIR


def _file_size_mb(path: Path) -> float:
    total = path.stat().st_size if path.exists() else 0
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / (1024 * 1024)


def export_v45_resampler_fp16(model_dir: str | Path, export_dir: str | Path) -> None:
    export_mod = importlib.import_module("01-Export-Vision-Encoder")
    model_dir = Path(model_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading V4.5 visual weights from {model_dir}...")
    model = export_mod.load_v45_model_for_export(model_dir)

    resampler_path = export_dir / "minicpmv_v45_resampler.fp16.onnx"
    print(f"\n[1/2] Exporting FP16 Resampler: {resampler_path}")
    wrapper = export_mod.ResamplerV45ONNX(model.resampler).half().eval()
    dummy = tuple(x.half() for x in export_mod.make_dummy_inputs_resampler_v45(1024, dtype=torch.float16))
    with torch.no_grad():
        out = wrapper(*dummy)
    print(f"  output: {list(out.shape)}")
    torch.onnx.export(
        wrapper,
        dummy,
        resampler_path,
        input_names=export_mod.RESAMPLER_V45_INPUT_NAMES,
        output_names=["visual_tokens"],
        dynamic_axes=export_mod.RESAMPLER_V45_DYNAMIC_AXES,
        opset_version=18,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  size: {_file_size_mb(resampler_path):.1f} MB")

    temporal_path = export_dir / "minicpmv_v45_resampler_temporal.fp16.onnx"
    print(f"\n[2/2] Exporting FP16 temporal Resampler: {temporal_path}")
    temporal_wrapper = export_mod.ResamplerTemporalV45ONNX(model.resampler).half().eval()
    temporal_dummy = tuple(
        x.half() for x in export_mod.make_dummy_inputs_resampler_temporal_v45(1024, dtype=torch.float16)
    )
    with torch.no_grad():
        temporal_out = temporal_wrapper(*temporal_dummy)
    print(f"  output: {list(temporal_out.shape)}")
    torch.onnx.export(
        temporal_wrapper,
        temporal_dummy,
        temporal_path,
        input_names=export_mod.RESAMPLER_TEMPORAL_V45_INPUT_NAMES,
        output_names=["visual_tokens"],
        dynamic_axes=export_mod.RESAMPLER_TEMPORAL_V45_DYNAMIC_AXES,
        opset_version=18,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  size: {_file_size_mb(temporal_path):.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FP16 MiniCPM-V 4.5 Resampler/3D-Resampler ONNX files. SigLIP remains FP32."
    )
    parser.add_argument("--model-dir", type=str, default=str(V45_MODEL_DIR))
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    args = parser.parse_args()

    export_v45_resampler_fp16(args.model_dir, args.export_dir)


if __name__ == "__main__":
    main()
