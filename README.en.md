<p align="center">
  <a href="README.md">中文版</a> | English
</p>

# MiniCPM-V-ONNX

Run local MiniCPM-V video understanding with an **ONNX Runtime vision encoder + official llama.cpp GGUF decoder**.

The LLM decoder stays in the official GGUF format. This repository exports only the vision encoder from the official Hugging Face safetensors checkpoint, then injects ONNX visual tokens into llama.cpp.

The llama.cpp server path is intentionally not used here: MiniCPM-V 4.5 video understanding needs the 3D-Resampler path, so the vision encoder is exported to ONNX and called directly.

## 1. Setup

```bash
git clone https://github.com/JazerJu/minicpm-v-onnx.git
cd minicpm-v-onnx

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get llama.cpp Runtime Libraries

This project calls llama.cpp's C API directly via ctypes. You need `.so` files in `bin/`.

### Option A: Download Pre-built Package (Recommended)

Download the tarball for your backend from [llama.cpp Release](https://github.com/ggml-org/llama.cpp/releases/latest):

| Backend | Download | Size |
|---------|----------|------|
| Vulkan (NVIDIA / AMD / Intel) | `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` | ~32 MB |
| CPU only | `llama-bXXXX-bin-ubuntu-x64.tar.gz` | ~15 MB |
| ROCm (AMD) | `llama-bXXXX-bin-ubuntu-rocm-x64.tar.gz` | ~128 MB |

> `bXXXX` is the llama.cpp build (version) number. The current Python ctypes bindings
> track the **b9409+** `llama_context_params` layout and have been validated locally
> with that newer runtime layout. If you replace the llama.cpp binaries, make sure
> `llama_model_params` / `llama_context_params` in `minicpmv_llama.py` still match
> your `llama.h`; a struct-layout mismatch can cause random initialization failures
> or C++ asserts.

Extract and copy `.so*` into `bin/`:

```bash
tar xzf llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz
cp -a llama-bXXXX-bin-ubuntu-vulkan-x64/lib*.so* bin/
```

`libvulkan.so` is a system package: `apt install libvulkan1` on Ubuntu / Debian.

### Option B: Build from Source

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# Vulkan (recommended, works on NVIDIA / AMD / Intel)
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release -j"$(nproc)"

# CPU only
# cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# CUDA
# cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

cp -a ../llama.cpp/build/bin/lib*.so* bin/
```

### Expected `bin/` Directory

```text
bin/
├── libllama.so*
├── libggml.so*
├── libggml-base.so*
├── libggml-cpu.so*
└── libggml-vulkan.so*   # or libggml-cuda.so* (depending on build backend)
```

Vulkan backend is recommended for NVIDIA users — best compatibility.

## 3. Download Models

Recommended video path, MiniCPM-V 4.5:

```bash
hf download openbmb/MiniCPM-V-4_5 \
  --local-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5

hf download openbmb/MiniCPM-V-4_5-gguf \
  --include "*Q4_K_M.gguf" \
  --local-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5_GGUF
```

MiniCPM-V 4.6, lighter (1.8B):

```bash
hf download openbmb/MiniCPM-V-4.6 \
  --local-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6

hf download openbmb/MiniCPM-V-4.6-gguf \
  --include "*Q4_K_M.gguf" \
  --local-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6_GGUF
```

## 4. Export V4.5 Video ONNX

MiniCPM-V 4.5 exports three runtime ONNX files: SigLIP, Resampler, and Temporal Resampler.

```bash
python3 01-Export-Vision-Encoder.py \
  --version 4.5 \
  --model-dir models/MiniCPM-V-4.5 \
  --export-dir models/MiniCPM-V-4.5/export \
  --example-h 8 \
  --example-w 8
```

Expected outputs:

```text
models/MiniCPM-V-4.5/export/
├── minicpmv_v45_siglip.fp32.onnx
├── minicpmv_v45_resampler.fp32.onnx
└── minicpmv_v45_resampler_temporal.fp32.onnx
```

Do not run `02-Optimize-Vision-Encoder.py` on V4.5 ONNX files. The V4.5 runtime path uses the unoptimized FP32 graphs.

## 5. Export V4.5 FP16 Resampler

`03-Quantize-Vision-Encoder.py` exports FP16 V4.5 Resampler / Temporal Resampler files. It does not touch SigLIP.

```bash
python3 03-Quantize-Vision-Encoder.py \
  --model-dir models/MiniCPM-V-4.5/MiniCPM-V-4_5 \
  --export-dir models/MiniCPM-V-4.5/export
```

Expected additional outputs:

```text
models/MiniCPM-V-4.5/export/
├── minicpmv_v45_resampler.fp16.onnx
└── minicpmv_v45_resampler_temporal.fp16.onnx
```

Video inference defaults to `minicpmv_v45_siglip.fp32.onnx` + `minicpmv_v45_resampler_temporal.fp16.onnx`.

## 6. Run Video

The repo includes a 20-second sample video `sample_video.mp4`:

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
  --question "Summarize this video." \
  --n-predict 256
```

For coarse summaries, reduce `--num-frames` to 7 or 8. For object or character presence checks in a short clip, use denser sampling.

> If you encounter cuDNN symbol errors with `--provider cuda`, see [experience/04_ld-library-path-cudnn.md](experience/04_ld-library-path-cudnn.md).

## 7. V4.6 Path

```bash
python3 01-Export-Vision-Encoder.py \
  --version 4.6 \
  --model-dir models/MiniCPM-V-4.6/MiniCPM-V-4_6 \
  --export-dir models/MiniCPM-V-4.5/export
```

`02-Optimize-Vision-Encoder.py` is only for the V4.6 single-file ONNX graph.

```bash
GGUF=$(find models/MiniCPM-V-4.6 -name '*Q4_K_M.gguf' | head -n 1)

python 04-Verify-Video.py sample_video.mp4 \
  --version 4.6 \
  --gguf "$GGUF" \
  --export-dir models/MiniCPM-V-4.5/export \
  --n-predict 256
```

## 8. Git-ignored Files

These paths are excluded by `.gitignore`:

- `models/` — original models and exported ONNX files
- `model/` — legacy path alias
- `bin/*.so*` — runtime libraries, users provide their own
- `vidUnder/` — separate video understanding project
- `OPENCODE_*` — IDE session files

## License

[MIT](LICENSE)
