# 3D-Resampler 架构与问题

## 总结

MiniCPM-V 4.5 的 3D-Resampler 负责将多帧视觉特征压缩为 64 个 token。在Onnx导出vision encoder的过程中经历了两次重大纠正：

1. **per-tile vs per-clip**：原以为是 per-tile（每 tile 产 64 token），后确认原版 PyTorch 是 per-clip（所有帧一次走 Resampler → 64 token），纠正这个概念后，需要调整不少参数，尤其是packing_size = 1，已写入agents.md，不能改动。
2. **BFC Arena 限制**：原以为是"8 tile / 21 帧是硬上限"，后确认根因是**输入 shape 不稳定**，不是帧数本身

## 架构概览(minicpm-v4.5的vision encode部分)

```
视频 → 帧提取 → 预处理(448×448 tiles)
       ↓
┌── siglip.fp32.onnx ──────────────────────────────────────────┐
│  SigLIP ViT (27层) per-tile                                  │
│  输入: pixel_values, pos_ids                                  │
│  输出: [N_patches, 1152]                                     │
└──────────────────────────────────────────────────────────────┘
       ↓
┌── resampler_temporal.fp32.onnx ──────────────────────────────┐
│  3D-Resampler per-clip                                       │
│  输入: siglip_features, spatial_pos_embeds, temporal_pos_embeds │
│  输出: [64, 4096]                                            │
└──────────────────────────────────────────────────────────────┘
       ↓
LLM Decoder (llama.cpp GGUF via ctypes)
```

## Resampler 内部流程

```
ONNX 外部传入（Python 计算）：
  siglip_features     [N_patches, 1152]  ← SigLIP per-tile 输出拼接
  spatial_pos_embeds  [N_patches, 4096]  ← preprocess.py get_2d_sincos_pos_embed_numpy
  temporal_pos_embeds [N_patches, 4096]  ← preprocess.py compute_temporal_embeddings_for_group
                                           （temporal_ids → sin-cos 编码）
```

### spatial_pos_embeds 与 temporal_pos_embeds

| | spatial_pos_embeds | temporal_pos_embeds |
|---|---|---|
| 编码维度 | **2D** sin-cos（row, col） | **1D** sin-cos（时间） |
| 粒度 | 每个 **patch 位置**一个唯一 embedding | 每**帧**一个 embedding，同帧所有 patch 共享（`np.tile`） |
| 来源 | `get_2d_sincos_pos_embed_numpy(embed_dim, (h, w))` | `encode_video_temporal_ids` → `get_1d_sincos_pos_embed_from_temporal` |
| 作用 | "这个 patch 在图像的哪个位置" | "这个 patch 属于哪一帧" |

**spatial**：同一张图里，左上角 patch 和右下角 patch 编码不同。同分辨率下每帧相同。

**temporal**：同帧的所有 patch 编码相同。帧 0 和帧 6 不同——这是跨帧建模时序关系的关键。

### 为什么 spatial_pos_embeds 在 Python 侧计算

SigLIP 和 Resampler 拆成两个独立 ONNX 后，Resampler ONNX 不再拥有图像尺寸信息，无法自己计算空间位置嵌入。因此 spatial_pos_embeds 和 temporal_pos_embeds 都在 Python 侧预计算好，作为 ONNX 输入传入。

**这恰好也帮助了 BFC Arena 稳定性**：因为 sin-cos 运算不在 ONNX 图内，ONNX 的输入 tensor shape 完全由 Python 侧控制。结合 `max_slice_nums=1` + `packing_size=1`，每次 ONNX 调用的输入维度始终是 1 帧的 patch 数（1024），shape 恒定，Arena 永不扩容。

```
ONNX 内部（固化在 .onnx 文件中的权重）：
         ↓
kv_proj: Linear(1152 → 4096) + ln_kv: LayerNorm
         ↓
+ spatial_pos_embeds + temporal_pos_embeds（加到 K 上）
         ↓
Cross-Attention: Q=64 个可学习 query token [64, 4096], K/V=[N_patches, 4096]
         ↓
ln_post: LayerNorm → proj: Linear(4096 → 4096)
         ↓
输出：[64, 4096]
```

三个输入全部由 Python 侧计算好后传入 ONNX。`temporal_ids` → `temporal_pos_embeds` 的 sin-cos 编码在 `preprocess.py` 完成，ONNX 内部不处理 temporal_ids。

**必须在 per-clip 模式下运行**——跨帧的 temporal_ids 才能让 Resampler 区分不同帧的 patch 并建模时序关系。

## Bug 日志

### ONNX导出拆分：从一个 ONNX 到三个 ONNX

**原计划**是仿照 V4.6 的做法——导出一个包含了 SigLIP ViT + Resampler 的**单文件 ONNX**。

代码中至今保留了首次尝试的 wrapper：`VisionEncoderV45ONNX`（`01-Export-Vision-Encoder.py` line 315），它的 `forward` 把 SigLIP 27 层 + post_layernorm + Resampler kv_proj/ln_kv/cross-attention/ln_post/proj 全串在一起，一次导出。

但这次发现碰了三个致命 bug：

```
VisionEncoderV45ONNX（首次尝试，单文件）
    ├── Bug #8: nn.LayerNorm + nn.Embedding 在 torchscript ONNX 导出时权重被重置为默认值
    │     → cosine 仅 0.97
    ├── Bug #5: do_constant_folding=True 融合权重腐蚀
    │     → cosine 暴跌到 0.48
    └── Bug #9: from_pretrained 不加载 vpm.* 权重
          → 视觉编码器整体噪声，端到端完全不对
```

**修复路径**：


```
Phase 1: SigLIPEasyV45ONNX（Bug #8 修复）
    ├── 手动 LayerNorm：_manual_layernorm(x, weight, bias) 替代 nn.LayerNorm
    ├── 手动 position lookup：pos_emb_w[pos_ids] 替代 nn.Embedding
    ├── 手动 Conv2d：register_buffer 显式保存权重
    └── 但不包含 Resampler —— 为了隔离问题，先把 SigLIP 单独导出

Phase 2: 手动 safetensors 加载 vpm.*（Bug #9 修复）
    └── from_pretrained 不加载 vpm. 前缀权重 → 从 safetensors 手动覆盖

Phase 3: 拆分 Resampler（ResamplerV45ONNX / ResamplerTemporalV45ONNX）
    ├── Resampler 本身没有 LayerNorm 问题（它的 LayerNorm 是 PyTorch 标准用法）
    ├── 但 cross-attention 需要手动拆解（_resampler_cross_attn）
    └── 两个变体：普通版（per-frame）+ temporal 版（per-clip，ONNX 输入比普通版多了 temporal_pos_embeds）
```

**最终文件**：

| ONNX 文件 | 大小 | 用途 |
|-----------|------|------|
| `minicpmv_v45_siglip.fp32.onnx` | 1.6 GB | SigLIP ViT (27层)，per-tile 推理 |
| `minicpmv_v45_resampler.fp32.onnx` | 340 MB | Resampler，per-frame（图消） |
| `minicpmv_v45_resampler_temporal.fp32.onnx` | 340 MB | Resampler，per-clip，输入含 temporal_pos_embeds |

**关键结论**：拆成三个 ONNX 不是架构设计——是 Bug #5/#8/#9 导致单文件导出不可行，不得不拆。后来发现拆分也有好处：SigLIP per-tile、Resampler per-clip 的解耦让输入 shape 控制更精确。

## SigLIP 为什么必须 FP32

### 架构差异

| | V4.6 | V4.5 |
|---|---|---|
| SigLIP 层数 | ~12 层 | **27 层** |
| 降维模块 | **vit_merger**（9216→1024，内置 ONNX） | **无**（输出 1152，靠外部 Resampler 降维） |
| FP16 可行性 | ✅ | ❌ |

### 发现过程（2026-06-07，msg_ea519d9ab）

V4.6 用 FP16 ONNX 成功后，自然对 V4.5 也尝试 FP16。

**SigLIP FP16 vs FP32 对比**（2026-06-07 04:21-04:24）：

```
对照对象：SigLIP 最终输出（post_layernorm 后的 hidden_states，即 ONNX 输出）
参考基准：PyTorch FP32 原始模型

SigLIP FP16 cosine vs FP32 参考 = 0.597 → 不可用
SigLIP FP32 cosine vs FP32 参考 = 0.999997 → 正常
```

**排查**：逐层对比 FP16 vs FP32 中间激活，发现误差在每层 attention 和 MLP 后逐步放大，27 层累积后输出向量方向已偏离。

**根因**：V4.6 的 vit_merger 在 ViT 输出后立即做 `Linear(9216→1024)`，这步矩阵乘法天然具有混合/平均效果，抑制了 FP16 逐层累积误差。V4.5 没有这一步——27 层 FP16 误差直接传导到 Resampler，无法挽回。

**对比 Resampler**（同日测试）：

```
Resampler FP16 cosine vs FP32 参考 = 0.999989 → 完全可用
```

Resampler 只有 6 层，模型小，FP16 误差不累积。

**修复**：SigLIP 固定 FP32（1.6GB），Resampler 可用 FP16（模型小、层数少、误差不累积，cosine 0.999989）。

| 组件 | 精度 | 原因 |
|------|------|------|
| SigLIP | **必须 FP32** | 27 层无降维，FP16 误差逐层累积 |
| Resampler | FP16 可选 | 模型小（6 层），FP16 误差可忽略 |
| Resampler Temporal | FP16 可选 | 同上 |

## packing_size 优化过程

### 阶段 1：packing_size=2（2026-05-30）

vidUnder build 阶段用 4 帧/clip，packing_size=2：

```
4 帧 → packing_size=2 → 2 组 × 64 = 128 visual tokens / clip
```

此时以为原版 PyTorch 也按 pair 分组。

### 阶段 2：原版行为纠正（2026-06-01）

原版 MiniCPM-V 4.5 PyTorch 的 Resampler 是所有帧一次传入，一次产 64 token，不分 pair。`packing_size` 是我们的适配方案，不是原版概念。

### 阶段 3：锁定 packing_size=1

```
packing_size=1 → 每帧一组 → N 帧 × 64 tokens → N×64 total（当前默认）
packing_size=N → 每组 N 帧 → 输入维度 N 倍，shape 随组变化
```

**为什么 packing_size=1**：

1. **输入 shape 稳定**：`max_slice_nums=1` 时每帧固定 1 tile，packing_size=1 保证每组输入维度恒定
2. **BFC Arena 教训**：之前错误认为"8 tile 是硬上限"（2026-06-07 纠正）。真正的问题不是帧数过多，而是**输入 shape 不稳定触发 Arena 重新分配或碎片化**。Resampler 本身可以处理 ~137K patches
3. **temporal_ids 明确**：每组 1 帧 = 每组 1 个 unique temporal_id，无歧义

## BFC Arena：从"硬上限"到"shape 不稳定"

### 错误结论的由来（2026-06-07 之前）

vidUnder build 阶段需要合并多个 clip 的帧统一走 Resampler。当合并帧数较多（14 帧、21 帧）时，ONNX Runtime 抛出 CUDA 分配失败。当时错误归因为"Resampler 单次限制约 8 tile（BFC Arena 2GB 硬上限）"，把这个数字写进了 AGENTS.md。

### 纠正（2026-06-07）

后续验证：Resampler 以 137,200 patches（≈21 帧）推理，cosine 正常、不崩溃。**帧数不是问题。**

真正触发崩溃的条件：同一 Resampler session 内，**前后两次调用的输入 shape 不一致**：

```
packing_size=2，4 帧：
  第一次 Resampler:  [2帧, 9800 patches] → Arena 分配 800 MB
  第二次 Resampler:  [2帧, 9800 patches] → 复用，正常 ✓

packing_size=2 vs 1 混用：
  第一次 Resampler:  [2帧, 9800 patches]  → Arena 分配 800 MB
  第二次 Resampler:  [4帧, 19600 patches] → 需要扩容到 1.5 GB
                                          → Arena 在 CUDA 显存碎片中扩容失败 ✗
```

ONNX Runtime CUDA BFC Arena 首次调用时按 shape 预分配，shape 不变则复用，shape 变化需扩容——扩容在碎片化显存中容易失败。这才是真正的原因，帧数本身不是问题。

### 结论

不是"8 tile / 21 帧硬上限"。根因是**输入 shape 切换导致 BFC Arena 扩容失败**。因此 packing_size 必须固定为 1——每组都是 1 帧的 tile 数，每次 Resampler 调用的输入 shape 恒定，Arena 始终复用。

## 管线验证历史（Bug #2 / #3）

**Bug #2** 确认了 per-tile SigLIP 可行（cosine=0.99997），为 SigLIP+Resampler 分拆架构提供了信心。

**Bug #3** 是关键前序依赖——ONNX 导出的 embedding 权重在初期没有被正确加载进计算图，导致 SigLIP 特征根本不对。这个 bug 修复后，Resampler 的输入才变得有意义。

## 测试

```bash
Mode: 3D-Resampler (packing=1)
7 frames, max_slice_nums=1, packing=1 → 7 packages × 64 = 448 visual tokens (6.0s)
All tests done. 1 passed.
```
