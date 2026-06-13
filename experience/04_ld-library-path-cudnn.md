# LD_LIBRARY_PATH 里的 cuDNN 版本会污染 ONNX Runtime CUDA

## 问题

同一台机器上有多个 Python/conda/venv 环境时，`LD_LIBRARY_PATH` 可能先指向别的环境里的 NVIDIA 动态库。

实际遇到的坏路径：

```text
~/anaconda3/envs/<other_env>/lib/python3.10/site-packages/nvidia/cudnn/lib
```

用 llama.cpp 的 `.venv` 跑 `04-Verify-Video.py --provider cuda` 时，ONNX Runtime CUDA 会先加载这个路径里的 cuDNN，导致 CUDA EP 初始化失败。

典型报错：

```text
Could not load symbol cudnnGetLibConfig.
Error: .../nvidia/cudnn/lib/libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig
```

## 根因

进程加载动态库时按 `LD_LIBRARY_PATH` 顺序搜索。即使 Python 解释器来自你的 venv（`.venv/bin/python`），只要 `LD_LIBRARY_PATH` 前面放了别的环境的 `nvidia/cudnn/lib`，ONNX Runtime 仍会加载错版本的 cuDNN。

这不是 GPU 不存在，也不是 CUDAExecutionProvider 没装；原因在动态库搜索顺序。

## 固定运行方式

运行 MiniCPM-V ONNX + llama.cpp 前，把当前 venv 的 NVIDIA 库路径放在最前面：

```bash
source .venv/bin/activate

LD_LIBRARY_PATH="$PWD/bin:$(python -c 'import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+\"/lib\")'):$(python -c 'import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__)+\"/lib\")'):$(python -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__)+\"/lib\")'):/usr/local/cuda/lib64" \
  python 04-Verify-Video.py sample_video.mp4 \
  --version 4.5 \
  --provider cuda \
  --num-frames 7 \
  --packing 1
```

## 排查命令

先看当前 shell 是否已经被别的环境污染：

```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
```

再确认当前 venv 里实际安装的 cuDNN/cuBLAS：

```bash
find "$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')" \
  -path '*cudnn/lib*' -o -path '*cublas/lib*'
```

## 规则

- 不要把 conda/autocut 等其他环境的 `nvidia/cudnn/lib` 放在本项目运行命令前面。
- `--provider cuda` 报 cuDNN symbol 错误时，先查 `LD_LIBRARY_PATH` 顺序。
- 如果 `CUDAExecutionProvider` 已经能被 `onnxruntime.get_available_providers()` 看到，但 session 初始化仍报 cuDNN symbol，优先按动态库污染处理。
