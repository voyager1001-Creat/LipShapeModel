# Training Data Format

Each subdirectory under this folder (e.g. `1/`) is one training sample and must contain a **same-name .npz** and **same-name .mp3**, for example:

- `1/1.npz` — Packed per-frame parameters and word–frame mapping
- `1/1.mp3` — Full audio (aligned with timestamps in the npz)

Training uses `WordFrameDataset` with e.g. `packed_npz_path=Project/data/1/1.npz`, `audio_path=Project/data/1/1.mp3`.

---

## 1. `*.npz` format (packed data)

Produced by packing “one directory per frame + word_frames.csv” (e.g. via `demo/project/data/pack_result_to_npz.py`).  
After `np.load(..., allow_pickle=True)` you get a dict with:

| Key | Shape/type | Description |
|-----|------------|-------------|
| `shape` | (300,) float32 | Face shape of the first frame, shared for the whole sequence |
| `exp` | (N, 100) float32 | Per-frame expression coefficients (FLAME 100-dim) |
| `jaw` | (N, 3) float32 | Per-frame jaw pose |
| `global_pose` | (N, 3) float32 | Per-frame global pose |
| `frame_names` | (N,) object | List of frame directory names, one-to-one with rows of exp/jaw/global_pose |
| `init_shape` | (300,) float32 | Same as `shape`, used as init |
| `init_exp` | (100,) float32 | First-frame exp |
| `init_jaw` | (3,) float32 | First-frame jaw |
| `init_global_pose` | (3,) float32 | First-frame global_pose |
| `word_frames_csv` | (1,) object | String: full text content of word_frames.csv |

- **N** = total number of frames (same as length of `frame_names`).  
- For chunked npz (e.g. `*_part0.npz`), there is an extra key `frame_offset` (int), the global start frame index of this chunk.

### `word_frames_csv` content (CSV)

One header row with columns: `word`, `start_sec`, `end_sec`, `start_frame`, `end_frame`, `frame_folders`.

| Column | Description |
|--------|-------------|
| word | Word text |
| start_sec / end_sec | Start/end time (seconds) of the word in the audio |
| start_frame / end_frame | Start/end frame indices (w.r.t. the N frames) |
| frame_folders | Frame directory names for this word, separated by `\|`, must match names in `frame_names` |

Example row:

```text
word,start_sec,end_sec,start_frame,end_frame,frame_folders
hello,0.12,0.48,4,14,000004_000|000005_000|...|000014_000
```

Training uses each row as “word + time + frame list”: `start_sec`/`end_sec` slice the audio; `frame_folders` index into the npz’s exp/jaw/global_pose to get GT (T, 106).

---

## 2. `*.mp3` format (audio)

- Same directory and base name as the npz (only extension differs), e.g. `1.mp3` with `1.npz`.
- Full-length audio; sample rate is not fixed; loading resamples to **16000 Hz** (via librosa, matching Wav2Vec2 / `AUDIO_SAMPLE_RATE` in config).
- `start_sec` / `end_sec` in the CSV must lie within this audio’s duration.

---

## 3. Directory layout example

```text
Project/data/
  training_data_format.md   (this file; 训练数据格式说明.md is the Chinese version)
  1/
    1.npz
    1.mp3
  2/
    2.npz
    2.mp3
```

For multiple sequences, the training script iterates over multiple `packed_npz_path` / `audio_path` pairs (or the dataset supports multiple directories).

---

## 4. How to generate npz + mp3 (obtaining training data)

Use the all-in-one script to produce **video →** `out_dir/<stem>.npz` and `out_dir/<stem>.mp3` (no large intermediate per-frame directories). Output names follow the video stem, e.g. `2.mp4` → `2.npz`, `2.mp3`; if a same-name .mp3 already exists next to the video, it is used and not copied.

- **Script**: `Project/scripts/video_to_training_data.py`
- **Run** from the **repo root**. If you see `No module named 'inferno.datasets'`, the repo is not installed as a package. Either:
  1. **Recommended**: from repo root run `pip install -e .` (editable install), then run `python Project/scripts/...`;
  2. Or run with env: `PYTHONPATH=. python Project/scripts/video_to_training_data.py ...`.

Example:

```bash
# Output same name as video: 2.mp4 → Project/data/2/2.npz and 2.mp3; uses existing 2.mp3 if present
python Project/scripts/video_to_training_data.py --video Project/data/2/2.mp4 --out_dir Project/data/2

# With a separate audio file (written to out_dir/xxx.mp3)
python Project/scripts/video_to_training_data.py --video 1.mp4 --audio 1.wav --out_dir Project/data/1
```

Common arguments:

| Argument | Description |
|----------|-------------|
| `--video` | Input video path (required) |
| `--audio` | Optional; if omitted, looks for same-name .mp3 next to video, else extracts from video and saves as `out_dir/<stem>.mp3` |
| `--out_dir` | Output directory; writes `out_dir/<stem>.npz` and `out_dir/<stem>.mp3` |
| `--path_to_models` | FaceReconstruction model dir; default from repo assets |
| `--model_name` | Model name; default `EMICA-CVT_flame2020_notexture` |
| `--fps` | Frame rate for word–frame alignment (default 30); may be overridden from video metadata |
| `--use_cpu` | Use CPU for everything |
| `--whisper_cpu` | Only Whisper on CPU (face still on GPU; use if you hit cuDNN errors) |
| `--use_modelscope` | Download Whisper from ModelScope (recommended in China) |
| `--whisper_model` | Whisper model: base / small / medium or local path |
| `--whisper_cache` | Whisper model dir; default `Project/model/whisper_models` |

The script extracts frames to a temp dir, runs face detection and FaceReconstruction to collect per-frame codes in memory, runs Whisper for word alignment, builds `word_frames_csv`, then writes the npz and mp3 and removes the temp dir.

---

## 5. Correspondence with FLAME / mouth dimensions

- Per-frame mouth parameters are **106-dim** = exp(100) + jaw(3) + global_pose(3).  
- The npz fields `exp`, `jaw`, `global_pose` match the semantics of `exp.npy`, `jawpose.npy`, `globalpose.npy` per frame in demo outputs (e.g. `init_frame0.npz`, `Result/3`), and can be used directly for training or FLAME mesh export (see `Project/emica_export`).
