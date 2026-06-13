# ONNX 输出名不要硬编码

## 问题

同一类 ONNX 图在不同导出/转换路径下，输出名可能不一致。

MiniCPM-V 4.5 temporal Resampler 已经实际遇到：

```text
minicpmv_v45_resampler_temporal.fp32.onnx  -> output: visual_tokens
minicpmv_v45_resampler_temporal.fp16.onnx  -> output: squeeze_1
```

如果推理代码写死：

```python
sess.run(["visual_tokens"], feed)
```

FP16 文件会直接报错：

```text
Invalid output name:visual_tokens
```

## 规则

推理代码不要硬编码 ONNX 输出名。加载 session 后读取真实输出名：

```python
output_name = sess.get_outputs()[0].name
result = sess.run([output_name], feed)[0]
```

输入名可以继续按导出约定固定，因为 feed 必须和图输入对应；输出名不是运行逻辑的一部分，不应该依赖它稳定。

## 适用位置

- `04-Verify-Video.py`
- 任何后续新增的 ONNX Runtime 推理脚本
