# Project：音频→口型 / Audio-to-Mouth

**中文：** 本目录为「初始 FLAME 参数 + 音频 + 文本 → 口型 106 维序列」的训练与推理流程，并可选导出 FLAME .obj 序列。训练与推理仅依赖本目录及 `requirements.txt`，不依赖 `demo/` 或 `inferno` 包；**仅** `scripts/video_to_training_data.py` 在生成训练数据时依赖 inferno。

**English:** This directory provides training and inference for **initial FLAME params + audio + text → 106-dim mouth sequence** (100 exp + 3 jaw + 3 global_pose), with optional FLAME .obj export. Training and inference depend only on this directory and `requirements.txt`, not on `demo/` or the `inferno` package; **only** `scripts/video_to_training_data.py` (video → training data) requires inferno.

---

## 运行方式 / How to Run

**中文：** 在 **inferno 仓库根** 执行（或将 Project 置于某目录并设该目录为 `PYTHONPATH`，且包名为 `Project`）。

**English:** Run from the **inferno repo root**, or put Project in a directory on `PYTHONPATH` with package name `Project`.

```bash
# 安装依赖 / Install deps
pip install -r Project/requirements.txt 

# 训练 / Train (needs Project/data/<id>/<id>.npz + <id>.mp3)
python Project/train_audio_to_mouth.py --data_root Project/data --checkpoint_dir Project/checkpoints/audio_text_to_mouth

# 推理 / Inference
python Project/infer_audio_to_npz.py --checkpoint Project/checkpoints/audio_text_to_mouth/best.pt --audio Project/data/2/2.mp3 --out_npz Project/data/2/2_pred.npz --init_npz Project/data/2/2.npz
```

---

## 获取训练数据 / Obtaining Training Data

**中文：** **获取数据需将本项目放在 inferno 仓库下**，在 **inferno 仓库根目录** 执行下方命令。脚本依赖 inferno 的人脸重建与检测。输出为 `out_dir/<视频名>.npz` 与 `.mp3`，即可用于训练。若只拷贝了 Project 而未放在 inferno 下，需按 `data/训练数据格式说明.md` 自行准备 npz + mp3。

**English:** **To obtain training data, place this Project inside the inferno repo** and run the commands below from the **inferno repo root**. The script depends on inferno’s face reconstruction and detection. Outputs are `out_dir/<stem>.npz` and `.mp3`, ready for training. If you only have the Project folder (no inferno), prepare npz + mp3 yourself per `data/training_data_format.md`.

```bash
# 安装 inferno / Install inferno (repo root)
pip install -e .

# 视频 → npz + mp3 / Video → npz + mp3
PYTHONPATH=. python Project/scripts/video_to_training_data.py \
  --video Project/data/8/8.mp4 \
  --audio Project/data/8/8.mp3 \
  --out_dir Project/data/8 \
  --whisper_cpu
```

---

## 目录结构 / Directory Structure

```
Project/
├── README.md                 # 本文件 / This file
├── audio_to_mouth_config.py  # 维度与训练配置 / Dims & training config
├── requirements.txt          # 最小依赖 / Minimal deps
├── train_audio_to_mouth.py   # 训练入口 / Train entry
├── infer_audio_to_npz.py     # 推理：音频→npz（可选 .obj）/ Inference
├── data/
│   └── 训练数据格式说明.md   # 数据格式 / Data format doc
├── model/
│   ├── audio_text_to_mouth.py
│   ├── whisper_models/       # 可选 / optional
│   ├── wav2vec2-base-960h/   # 可选 / optional
│   └── FaceReconstruction/   # 仅 video_to_training_data 用 / for video script only
├── emica_export/             # 106 维 → FLAME → .obj（可独立）/ Standalone mesh export
├── scripts/
│   ├── video_to_training_data.py  # 视频→npz+mp3（依赖 inferno）/ Video→data (needs inferno)
│   └── inspect_npz.py
└── checkpoints/              # 训练输出 / Train output
```

---

## model 目录下需放置的模型 / Models to Place Under `model/`

**中文：** 以下目录/文件**不随仓库上传**，使用前请自行下载或从 inferno 拷贝到 **`Project/model/`** 下对应位置：

| 路径（置于 model/ 下） | 用途 |
|------------------------|------|
| `FLAME/` | FLAME 几何与关键点嵌入，供 `emica_export` 导出 .obj；需包含 `geometry/generic_model.pkl` 等（见 `emica_export/README.md`） |
| `whisper_models/` | Whisper / faster-whisper 模型，供推理时词对齐与训练数据脚本使用 |
| `wav2vec2-base-960h/` | Wav2Vec2 音频编码器（或由 transformers 自动下载），供训练与推理 |
| `FaceReconstruction/` | 仅 `scripts/video_to_training_data.py`（视频→训练数据）需要，为人脸重建模型资源 |

**English:** The following are **not included in the repo**; place them under **`Project/model/`** before use:

| Path (under model/) | Purpose |
|--------------------|---------|
| `FLAME/` | FLAME geometry and landmark embeddings for `emica_export` (.obj); include `geometry/generic_model.pkl` etc. (see `emica_export/README.md`) |
| `whisper_models/` | Whisper / faster-whisper models for word alignment (inference and video→data script) |
| `wav2vec2-base-960h/` | Wav2Vec2 audio encoder (or auto-downloaded by transformers) for training and inference |
| `FaceReconstruction/` | Only needed by `scripts/video_to_training_data.py` (video → training data); face reconstruction assets |

---

## 依赖说明 / Dependencies

**中文：**

- **requirements.txt**：仅列出 `torch`、`numpy`、`chumpy`（emica_export 用）。训练/推理实际还需：`transformers`、`librosa` 或 `soundfile`、`faster-whisper` 或 `openai-whisper` 等（可由 inferno 或本机环境提供）。
- **训练/推理**：使用 Project 内实现，无 `demo`/`inferno` 依赖。
- **视频→训练数据**：`video_to_training_data.py` 依赖 `inferno`/`inferno_apps`，需在完整 inferno 仓库内运行。

**English:**

- **requirements.txt** lists only `torch`, `numpy`, `chumpy` (for emica_export). Training/inference also need `transformers`, `librosa` or `soundfile`, `faster-whisper` or `openai-whisper`, etc. (from inferno or your environment).
- **Training/inference**: implemented inside Project, no `demo`/`inferno` dependency.
- **Video → training data**: `video_to_training_data.py` depends on `inferno`/`inferno_apps`; run inside the full inferno repo.

---

## 致谢 / Acknowledgments

**中文：** 本 Project 的「视频→训练数据」流程及 FLAME 相关实现参考并依赖 [inferno](https://github.com/famle/inferno) 项目的人脸重建、数据模块与资源，特此致谢。

**English:** The “video → training data” pipeline and FLAME-related code in this Project rely on and are inspired by the [inferno](https://github.com/famle/inferno) project (face reconstruction, data modules, and assets). We thank the inferno project and its contributors.
