# AudioTextToMouth 网络结构说明与合理性检查

## 1. 整体数据流

```
输入:
  shape      (B, 300)   FLAME identity
  exp        (B, 100)   首帧表情
  jaw        (B, 3)     首帧下颌
  global_pose(B, 3)      首帧全局姿态
  raw_audio  (B, 1, L)  该词对应音频片段（16kHz）
  word_id    (B,)        词表 id
  mask       (B, T)      有效帧掩码

输出:
  (B, T, 106)  每帧口型：exp(100) + jaw(3) + global_pose(3)
```

- **训练**：每个样本是一「词」→ 一段音频 + 对应 T 帧 GT，batch 内 T 取 max_T，不足的用 mask 标 0，loss 只对 mask=1 做 MSE。
- **推理**：按词逐段推理，再按帧写回全局序列并做重叠帧平均。

---

## 2. 维度与模块

| 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|
| shape_mlp | (B, 300) | (B, 128) | 300→256→128，identity 编码 |
| init_mlp | (B, 106) | (B, 64) | 106→128→64，首帧姿态编码 |
| text_embed | (B,) word_id | (B, 64) | Embedding(vocab, 64), padding_idx=1 |
| audio_encoder | raw_audio + desired_output_length=T | (B, T, 768) | Wav2Vec2 + 时间重采样到 T |
| fusion_proj | (B, T, 128+64+64+768)=1024 | (B, T, 256) | 四路 concat 后线性 |
| decoder | (B, T, 256) | (B, T, 256) | 2 层 TransformerEncoder, nhead=4, ff=512 |
| output_head | (B, T, 256) | (B, T, 106) | 单层 Linear |

- **Fusion 输入**：z_id(128) + z_init(64) + z_txt(64) + audio(768) = 1024 → 256，维度一致。
- **Wav2Vec2**：target_fps=30、expected_fps=50，与 30fps 口型对齐；若输出长度与 T 不一致，forward 内会再线性插值到 T。

---

## 3. 合理性结论

- **任务匹配**：词级片段 → 该词对应 T 帧口型，输入输出一一对应，设计合理。
- **多模态**：identity + 首帧姿态 + 词嵌入 + 音频，四路在时间维复制/对齐后融合，再 Transformer 解码到 106 维，结构常见且合理。
- **词信息**：同一词在同一片段内共享一个 embedding、复制到所有 T，语义一致，合理。
- **训练/推理**：训练用 mask 做 padding、推理用 desired_output_length，两路都保证 T 有定义；forward 已对「无 mask 且无 desired_output_length」做报错防护。
- **Loss**：仅对有效帧做 MSE，与 mask 一致，合理。

---

## 4. 可选改进（非必须）

- **Decoder**：当前为 2 层、nhead=4、dim_feedforward=512；数据多时可尝试加深/加宽或加 dropout。
- **Init 来源**：目前首帧姿态来自 npz 的 init_*；若希望更强时序一致，可考虑用上一段最后一帧作为下一段 init（需改数据流）。
- **正则**：未对 106 维做平滑或范围约束，若口型抖动可加 temporal smoothness 或 range clamp。

总体：**当前网络模型设计合理，维度与训练/推理流程一致，可直接使用。**
