# coding=utf-8
"""Legacy optional optimizer for MiniCPM-V 4.6 single-file ONNX only.

Do not run this script on MiniCPM-V 4.5 runtime ONNX files.  The V4.5 stable
pipeline relies on the exact FP32 graphs exported by `01-Export-Vision-Encoder.py`
(`minicpmv_v45_siglip.fp32.onnx`,
`minicpmv_v45_resampler.fp32.onnx`,
`minicpmv_v45_resampler_temporal.fp32.onnx`) and uses
`GraphOptimizationLevel.ORT_DISABLE_ALL` at runtime.
"""
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import onnx
from onnxruntime.transformers.optimizer import optimize_model

sys.path.insert(0, str(Path(__file__).parent))
from export_config import DEFAULT_EXPORT_DIR


def optimize_onnx(input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    print(f"Optimizing: {input_path}")

    optimizer = optimize_model(
        input_path,
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        opt_level=1,
    )

    model = optimizer.model
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
    )
    print(f"Saved: {output_path}")

    saved = onnx.load(output_path, load_external_data=False)

    print("Opset versions:")
    for imp in saved.opset_import:
        domain = imp.domain if imp.domain else "ai.onnx"
        print(f"  {domain}: {imp.version}")

    domain_ops = defaultdict(set)
    for node in saved.graph.node:
        domain = node.domain if node.domain else "ai.onnx"
        domain_ops[domain].add(node.op_type)

    print("Operator distribution:")
    for domain, ops in sorted(domain_ops.items()):
        print(f"  [{domain}] {', '.join(sorted(ops))}")

    graph_mb = os.path.getsize(output_path) / (1024 * 1024)
    data_path = output_path + ".data"
    data_mb = os.path.getsize(data_path) / (1024 * 1024) if os.path.exists(data_path) else 0
    print(f"Size: {graph_mb:.1f} MB (graph) + {data_mb:.1f} MB (weights)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optional legacy optimizer for MiniCPM-V 4.6 minicpmv_vision_encoder.fp32.onnx only. Do not use for V4.5."
    )
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    args = parser.parse_args()

    onnx_path = os.path.join(args.export_dir, "minicpmv_vision_encoder.fp32.onnx")
    if not os.path.exists(onnx_path):
        print(f"Not found: {onnx_path}")
        sys.exit(1)

    optimize_onnx(onnx_path)
