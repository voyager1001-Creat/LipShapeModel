"""
按 word_frames.csv 的 PyTorch Dataset：每样本为 (init, 音频片段, word_id, GT (T,106), T/mask)。
支持从「打包 npz」加载（与 video_to_training_data / pack_result_to_npz 生成格式一致）。
独立于 demo，仅依赖 Project.audio_to_mouth_config。
"""
import csv
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..audio_to_mouth_config import (
    AUDIO_SAMPLE_RATE,
    N_EXP,
    N_JAW,
    N_GLOBAL_POSE,
    OUTPUT_DIM,
)


def build_vocab_from_csv(csv_path: Path):
    """从 word_frames.csv 的 word 列统计词表，返回 word2id (含 <unk>=0, <pad>=1)。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        return build_vocab_from_csv_string(f.read())


def build_vocab_from_csv_string(csv_text: str):
    """从 CSV 字符串统计词表（用于从打包 npz 内的 word_frames_csv 建表）。"""
    word2id = {"<unk>": 0, "<pad>": 1}
    r = csv.DictReader(io.StringIO(csv_text))
    for row in r:
        w = row.get("word", "").strip()
        if w and w not in word2id:
            word2id[w] = len(word2id)
    return word2id


def load_audio_segment(
    audio_path: Path, start_sec: float, end_sec: float, sr: int = AUDIO_SAMPLE_RATE
) -> np.ndarray:
    """加载音频并在 [start_sec, end_sec] 截取，重采样到 sr。返回 (samples,) float32。"""
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for audio loading. pip install librosa")
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    start_samp = int(round(start_sec * sr))
    end_samp = int(round(end_sec * sr))
    start_samp = max(0, min(start_samp, len(y)))
    end_samp = max(start_samp, min(end_samp, len(y)))
    return y[start_samp:end_samp].astype(np.float32)


def _is_valid_word_frame_row(row: dict) -> bool:
    """CSV 行需有可解析的 start_sec、end_sec 才参与训练。"""
    try:
        s = row.get("start_sec", "").strip()
        e = row.get("end_sec", "").strip()
        if not s or not e:
            return False
        float(s)
        float(e)
        return True
    except (ValueError, TypeError):
        return False


def _segment_from_full_audio(
    full_audio: np.ndarray, start_sec: float, end_sec: float, sr: int
) -> np.ndarray:
    """从已加载的整段音频中截取 [start_sec, end_sec]，返回 (samples,) float32。"""
    start_samp = int(round(start_sec * sr))
    end_samp = int(round(end_sec * sr))
    start_samp = max(0, min(start_samp, len(full_audio)))
    end_samp = max(start_samp, min(end_samp, len(full_audio)))
    return full_audio[start_samp:end_samp].astype(np.float32)


class WordFrameDataset(Dataset):
    """
    每样本：init (shape, exp, jaw, global_pose)、音频波形、word_id、GT (T, 106)、帧数 T。
    若提供 packed_npz_path，则从打包 npz 读帧数据与 CSV；否则需 result_dir + init_npz_path + csv_path。
    """

    def __init__(
        self,
        csv_path: Path = None,
        result_dir: Path = None,
        init_npz_path: Path = None,
        audio_path: Path = None,
        word2id: dict = None,
        indices: list = None,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        packed_npz_path: Path = None,
    ):
        self.audio_path = Path(audio_path) if audio_path else None
        self.word2id = word2id
        self.sample_rate = sample_rate
        self.packed_npz_path = Path(packed_npz_path) if packed_npz_path else None
        self._full_audio = None

        if self.packed_npz_path is not None and self.packed_npz_path.is_file():
            self._init_from_packed()
            if indices is not None:
                self.rows = [self.rows[i] for i in indices]
            self._load_full_audio_once()
            return

        self.result_dir = Path(result_dir)
        self._frame_name_to_idx = None
        self.init_npz_path = Path(init_npz_path)
        init_data = dict(np.load(self.init_npz_path, allow_pickle=False))
        self.init_shape = np.asarray(init_data["shape"], dtype=np.float32)
        self.init_exp = np.asarray(init_data["exp"], dtype=np.float32)
        self.init_jaw = np.asarray(init_data["jaw"], dtype=np.float32)
        self.init_global_pose = np.asarray(init_data["global_pose"], dtype=np.float32)

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if _is_valid_word_frame_row(row):
                    rows.append(row)
        if indices is not None:
            rows = [rows[i] for i in indices]
        self.rows = rows
        self._load_full_audio_once()

    def _load_full_audio_once(self):
        """只加载一次整段音频到内存。"""
        if self.audio_path is None or not Path(self.audio_path).exists():
            return
        try:
            import librosa
        except ImportError:
            return
        self._full_audio, _ = librosa.load(
            str(self.audio_path), sr=self.sample_rate, mono=True
        )

    def _init_from_packed(self):
        """从打包 npz 加载（与 video_to_training_data / pack_result_to_npz 格式一致）。"""
        data = dict(np.load(self.packed_npz_path, allow_pickle=True))
        self.init_shape = np.asarray(data["init_shape"], dtype=np.float32)
        self.init_exp = np.asarray(data["init_exp"], dtype=np.float32)
        self.init_jaw = np.asarray(data["init_jaw"], dtype=np.float32)
        self.init_global_pose = np.asarray(data["init_global_pose"], dtype=np.float32)
        self._exp = np.asarray(data["exp"], dtype=np.float32)
        self._jaw = np.asarray(data["jaw"], dtype=np.float32)
        self._global_pose = np.asarray(data["global_pose"], dtype=np.float32)
        # 分块 npz 时有 frame_offset，表示本块在全局中的起始帧下标（见训练数据格式说明）
        self._frame_offset = int(data.get("frame_offset", 0))
        frame_names = data["frame_names"]
        if hasattr(frame_names, "tolist"):
            frame_names = frame_names.tolist()
        self._frame_name_to_idx = {str(n).strip(): i for i, n in enumerate(frame_names)}
        csv_text = data["word_frames_csv"]
        if hasattr(csv_text, "item"):
            csv_text = csv_text.item()
        all_rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.rows = [r for r in all_rows if _is_valid_word_frame_row(r)]
        if self.word2id is None:
            self.word2id = build_vocab_from_csv_string(csv_text)
        self.result_dir = None
        self.init_npz_path = None

    def _get_audio_segment(self, start_sec: float, end_sec: float) -> np.ndarray:
        if self._full_audio is not None:
            return _segment_from_full_audio(
                self._full_audio, start_sec, end_sec, self.sample_rate
            )
        return load_audio_segment(
            self.audio_path, start_sec, end_sec, self.sample_rate
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        word = row.get("word", "").strip()
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        frame_folders = (
            row.get("frame_folders", "").strip().split("|")
            if row.get("frame_folders")
            else []
        )
        frame_folders = [f.strip() for f in frame_folders if f.strip()]

        wave = self._get_audio_segment(start_sec, end_sec)
        word_id = self.word2id.get(word, self.word2id["<unk>"])

        # 该词的初始姿态：用「该词起始前一帧」的口型，而不是整段视频的全局 init
        # start_frame/end_frame 为 CSV 中的起止帧号（与 npz 中 N 帧对应；分块时为全局下标；可能是小数需取整）
        if getattr(self, "_exp", None) is not None and "start_frame" in row:
            try:
                start_frame = int(round(float(row["start_frame"])))
            except (ValueError, TypeError):
                start_frame = 0
            frame_offset = getattr(self, "_frame_offset", 0)
            prev_global = start_frame - 1
            prev_local = prev_global - frame_offset
            n_frames = len(self._exp)
            if start_frame > 0 and 0 <= prev_local < n_frames:
                init_exp_this = self._exp[prev_local].copy()
                init_jaw_this = self._jaw[prev_local].copy()
                init_global_pose_this = self._global_pose[prev_local].copy()
            else:
                init_exp_this = self.init_exp.copy()
                init_jaw_this = self.init_jaw.copy()
                init_global_pose_this = self.init_global_pose.copy()
        else:
            init_exp_this = self.init_exp.copy()
            init_jaw_this = self.init_jaw.copy()
            init_global_pose_this = self.init_global_pose.copy()

        T = len(frame_folders)
        gt = np.zeros((T, OUTPUT_DIM), dtype=np.float32)
        if self._frame_name_to_idx is not None:
            for t, folder in enumerate(frame_folders):
                i = self._frame_name_to_idx.get(folder)
                if i is not None:
                    gt[t, :N_EXP] = self._exp[i]
                    gt[t, N_EXP : N_EXP + N_JAW] = self._jaw[i]
                    gt[t, N_EXP + N_JAW :] = self._global_pose[i]
        else:
            for t, folder in enumerate(frame_folders):
                d = self.result_dir / folder
                if (
                    (d / "exp.npy").exists()
                    and (d / "jawpose.npy").exists()
                    and (d / "globalpose.npy").exists()
                ):
                    gt[t, :N_EXP] = np.load(d / "exp.npy")
                    gt[t, N_EXP : N_EXP + N_JAW] = np.load(d / "jawpose.npy")
                    gt[t, N_EXP + N_JAW :] = np.load(d / "globalpose.npy")

        return {
            "shape": self.init_shape.copy(),
            "exp": init_exp_this,
            "jaw": init_jaw_this,
            "global_pose": init_global_pose_this,
            "raw_audio": wave,
            "word_id": word_id,
            "gt": gt,
            "T": T,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }


def collate_word_frames(batch):
    """
    Padding：音频按 batch 内最大长度 pad；GT 按 max_T pad；返回 mask (B, T)。
    raw_audio 输出 (B, 1, L)，samplerate 为列表，供 Wav2Vec2 使用。
    """
    B = len(batch)
    max_T = max(b["T"] for b in batch)
    max_audio_len = max(len(b["raw_audio"]) for b in batch)

    shapes = np.stack([b["shape"] for b in batch])
    exps = np.stack([b["exp"] for b in batch])
    jaws = np.stack([b["jaw"] for b in batch])
    global_poses = np.stack([b["global_pose"] for b in batch])

    padded_audio = np.zeros((B, max_audio_len), dtype=np.float32)
    for i, b in enumerate(batch):
        L = len(b["raw_audio"])
        padded_audio[i, :L] = b["raw_audio"]
    raw_audio = torch.from_numpy(padded_audio).unsqueeze(1)

    word_ids = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
    T_list = [b["T"] for b in batch]
    mask = torch.zeros(B, max_T, dtype=torch.float32)
    gt_padded = torch.zeros(B, max_T, OUTPUT_DIM, dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["T"]
        mask[i, :T] = 1.0
        gt_padded[i, :T] = torch.from_numpy(b["gt"])

    return {
        "shape": torch.from_numpy(shapes).float(),
        "exp": torch.from_numpy(exps).float(),
        "jaw": torch.from_numpy(jaws).float(),
        "global_pose": torch.from_numpy(global_poses).float(),
        "raw_audio": raw_audio,
        "samplerate": [AUDIO_SAMPLE_RATE] * B,
        "word_id": word_ids,
        "gt": gt_padded,
        "mask": mask,
        "T": max_T,
        "lengths": T_list,
    }
