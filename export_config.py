# coding=utf-8
"""
MiniCPM-V ONNX Export Pipeline 配置
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
EXPORT_DIR = MODELS_DIR / "MiniCPM-V-4.5" / "export"

# V4.6 (1.3B, safetensors 单文件)
V46_MODEL_DIR = MODELS_DIR / "MiniCPM-V-4.6" / "MiniCPM-V-4_6"
V46_LLM_GGUF = MODELS_DIR / "MiniCPM-V-4.6" / "MiniCPM-V-4_6_GGUF" / "MiniCPM-V-4_6-Q4_K_M.gguf"
V46_MMPROJ_GGUF = MODELS_DIR / "MiniCPM-V-4.6" / "MiniCPM-V-4_6_GGUF" / "mmproj-model-f16-v46.gguf"

# V4.5 (8.7B, safetensors 4 分片)
V45_MODEL_DIR = MODELS_DIR / "MiniCPM-V-4.5" / "MiniCPM-V-4_5"
V45_LLM_GGUF = MODELS_DIR / "MiniCPM-V-4.5" / "MiniCPM-V-4_5_GGUF" / "MiniCPM-V-4_5-Q4_K_M.gguf"
V45_MMPROJ_GGUF = MODELS_DIR / "MiniCPM-V-4.5" / "MiniCPM-V-4_5_GGUF" / "mmproj-model-f16-v45.gguf"

# 默认导出 V4.6
DEFAULT_MODEL_DIR = V46_MODEL_DIR
DEFAULT_EXPORT_DIR = EXPORT_DIR
