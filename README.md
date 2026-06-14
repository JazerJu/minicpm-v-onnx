<p align="center">
  中文版 | <a href="README.en.md">English</a>
</p>

# MiniCPM-V-ONNX

让 MiniCPM-V 用 **ONNX Runtime 视觉编码器 + llama.cpp 官方 GGUF decoder** 跑本地视频理解。

MiniCPM-V 的 decoder 已经有官方 GGUF，所以这里不转换 LLM decoder。仓库只负责从官方 Hugging Face safetensors 导出视觉 encoder ONNX，然后把 ONNX 视觉 token 交给 llama.cpp 继续生成文本。

> llama.cpp 官方 server 只支持 MiniCPM 多图输入，不支持 V4.5 视频路径里的 3D-Resampler 视觉 token 压缩。因此完整视频理解需要导出视觉 encoder ONNX。

适合的场景：

- 本地视频理解
- 视频内容检索、问答和摘要
- 给自己的 AI agent 增加视频输入能力
- 复用官方 GGUF decoder，但不让 Transformers/PyTorch 常驻推理

## 1. 准备环境

```bash
git clone https://github.com/JazerJu/minicpm-v-onnx.git
cd minicpm-v-onnx

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 获取 llama.cpp 运行库

本项目通过 ctypes 直接调用 llama.cpp 的 C API，需要把 `.so` 文件放入 `bin/`。

### 方式一：下载预编译包（推荐）

从 [llama.cpp Release](https://github.com/ggml-org/llama.cpp/releases/latest) 下载对应后端的 tarball：

| 后端 | 下载文件 | 大小 |
|------|---------|------|
| Vulkan（NVIDIA / AMD / Intel） | `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` | ~32 MB |
| CPU only | `llama-bXXXX-bin-ubuntu-x64.tar.gz` | ~15 MB |
| ROCm (AMD) | `llama-bXXXX-bin-ubuntu-rocm-x64.tar.gz` | ~128 MB |

> `bXXXX` 是 llama.cpp 的 build（版本）号。本项目已测试 **b9159**（最低 **b7668**），请在此范围内选择 release。

解压后把 `.so*` 复制到 `bin/`：

```bash
tar xzf llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz
cp -a llama-bXXXX-bin-ubuntu-vulkan-x64/lib*.so* bin/
```

`libvulkan.so` 是系统包，Ubuntu / Debian 用 `apt install libvulkan1` 安装。

### 方式二：从源码编译

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# Vulkan（推荐，NVIDIA / AMD / Intel 通用）
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release -j"$(nproc)"

# CPU only
# cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# CUDA
# cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

cp -a ../llama.cpp/build/bin/lib*.so* bin/
```

### 最终 `bin/` 目录

```
bin/
├── libllama.so*
├── libggml.so*
├── libggml-base.so*
├── libggml-cpu.so*
└── libggml-vulkan.so*   # 或 libggml-cuda.so*（取决于编译后端）
```

NVIDIA 用户推荐 Vulkan 后端，兼容性最好。

## 3. 下载官方模型

推荐使用 MiniCPM-V 4.5 跑视频：

```bash
hf download openbmb/MiniCPM-V-4_5 \
  --local-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5

hf download openbmb/MiniCPM-V-4_5-gguf \
  --include "*Q4_K_M.gguf" \
  --local-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5_GGUF
```

V4.6 较为轻量（1.8B）：

```bash
hf download openbmb/MiniCPM-V-4.6 \
  --local-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6

hf download openbmb/MiniCPM-V-4.6-gguf \
  --include "*Q4_K_M.gguf" \
  --local-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6_GGUF
```

## 4. 导出 V4.5 视频 ONNX

MiniCPM-V 4.5 视频路径需要导出 SigLIP、普通 Resampler、Temporal Resampler 三个 ONNX 文件：

```bash
python3 01-Export-Vision-Encoder.py \
  --version 4.5 \
  --model-dir models/MiniCPM-V-4.5 \
  --export-dir models/MiniCPM-V-4.5/export \
  --example-h 8 \
  --example-w 8
```

导出后应得到：

```text
models/MiniCPM-V-4.5/export/
├── minicpmv_v45_siglip.fp32.onnx
├── minicpmv_v45_resampler.fp32.onnx
└── minicpmv_v45_resampler_temporal.fp32.onnx
```

不要把 `02-Optimize-Vision-Encoder.py` 用在 V4.5 的 ONNX 上；V4.5 稳定路径依赖未优化的 FP32 图。

## 5. 导出 V4.5 FP16 Resampler

`03-Quantize-Vision-Encoder.py` 只处理 V4.5 Resampler / Temporal Resampler，不处理 SigLIP：

```bash
python3 03-Quantize-Vision-Encoder.py \
  --model-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5 \
  --export-dir models/MiniCPM-V-4.5/export
```

导出后会增加：

```text
models/MiniCPM-V-4.5/export/
├── minicpmv_v45_resampler.fp16.onnx
└── minicpmv_v45_resampler_temporal.fp16.onnx
```

运行视频默认使用 `minicpmv_v45_siglip.fp32.onnx` + `minicpmv_v45_resampler_temporal.fp16.onnx`。

## 6. 跑视频

仓库附带一个 20 秒示例视频 `sample_video.mp4`：

```bash
source .venv/bin/activate
GGUF=$(find models/MiniCPM-V-4.5 -name '*Q4_K_M.gguf' | head -n 1)

python 04-Verify-Video.py sample_video.mp4 \
  --version 4.5 \
  --gguf "$GGUF" \
  --export-dir models/MiniCPM-V-4.5/export \
  --provider cuda \
  --num-frames 140 \
  --packing 1 \
  --resampler-dtype fp16 \
  --ctx-size 12288 \
  --question "请总结这个 20 秒视频的主要内容。" \
  --n-predict 256
```

`--num-frames 140` 适合 20 秒视频里的目标检索问题；如果只做粗略摘要，可以降到 7 或 8。

> 使用 CUDA `--provider cuda` 时，如果遇到 cuDNN symbol 错误，请参考 [experience/04_ld-library-path-cudnn.md](experience/04_ld-library-path-cudnn.md)。

## 7. V4.6 轻量视频路径

V4.6 可以导出单文件视觉 encoder：

```bash
python3 01-Export-Vision-Encoder.py \
  --version 4.6 \
  --model-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6 \
  --export-dir models/MiniCPM-V-4.5/export
```

`02-Optimize-Vision-Encoder.py` 仅用于 V4.6 单文件 ONNX，不用于 V4.5 分体 ONNX。

V4.6 视频示例：

```bash
GGUF=$(find models/MiniCPM-V-4.6 -name '*Q4_K_M.gguf' | head -n 1)

python 04-Verify-Video.py sample_video.mp4 \
  --version 4.6 \
  --gguf "$GGUF" \
  --export-dir models/MiniCPM-V-4.5/export \
  --n-predict 256
```

## 8. 不进入仓库的文件

以下文件默认被 `.gitignore` 排除：

- `models/` — 原始模型和导出文件
- `model/` — 兼容旧路径
- `bin/*.so*` — 运行库，用户自行获取
- `vidUnder/` — 独立的视频理解项目
- `OPENCODE_*` — IDE 会话文件

## License

[MIT](LICENSE)
