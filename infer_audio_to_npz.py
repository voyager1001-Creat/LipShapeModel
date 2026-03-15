"""
推理：音频 + Whisper 对齐 → 预测口型序列 → 写出 npz（与 Project/data/<id>.npz 相同格式），可选导出 FLAME .obj 序列。

示例（在仓库根）：
  python Project/infer_audio_to_npz.py --checkpoint Project/checkpoints/audio_text_to_mouth/best.pt \\
    --audio Project/data/2/2.mp3 --out_npz Project/data/2/2_pred.npz --init_npz Project/data/2/2.npz --whisper_cpu
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch

# 仓库根（Project 的上一级）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Project.whisper_align import run_whisper  # noqa: E402
from Project.audio_to_mouth_config import (  # noqa: E402
    AUDIO_SAMPLE_RATE,
    N_EXP,
    N_GLOBAL_POSE,
    N_JAW,
    OUTPUT_DIM,
    TARGET_FPS,
)
from Project.data import load_audio_segment  # noqa: E402
from Project.model.audio_text_to_mouth import AudioTextToMouth  # noqa: E402
from Project.emica_export import export_sequence_to_obj  # noqa: E402


def _build_word_frames_and_frames_from_audio(
    word_triples,
    fps: float,
) -> tuple[list[str], str]:
    """
    根据 Whisper 词时间戳 + fps：
    - 决定总帧数 N（由 max end_sec * fps 决定）
    - 生成 frame_names: ['000001_000', ..., '00NNNN_000']
    - 生成 word_frames_csv 字符串（与训练阶段相同格式）
    """
    if not word_triples:
        raise ValueError("Whisper 未返回任何词级时间戳")
    max_end = max(float(e) for _, _, e in word_triples)
    num_frames = max(1, int(round(max_end * fps)) + 1)
    frame_names = [f"{i+1:06d}_000" for i in range(num_frames)]

    lines = ["word,start_sec,end_sec,start_frame,end_frame,frame_folders"]
    for word, start_sec, end_sec in word_triples:
        start_sec = float(start_sec)
        end_sec = float(end_sec)
        start_frame = max(0, min(round(start_sec * fps), num_frames - 1))
        end_frame = max(0, min(round(end_sec * fps), num_frames - 1))
        if start_frame > end_frame:
            end_frame = start_frame
        folders = frame_names[start_frame : end_frame + 1]
        lines.append(
            f"{word},{round(start_sec, 4)},{round(end_sec, 4)},{start_frame},{end_frame},"
            f"{'|'.join(folders)}"
        )
    csv_text = "\n".join(lines)
    return frame_names, csv_text


def _load_or_build_word_frames_csv(
    audio_path: Path,
    fps: float,
    whisper_model: str,
    whisper_cache: Path | None,
    use_modelscope: bool,
    whisper_cpu: bool,
    no_ssl_verify: bool,
    csv_path: Path | None,
) -> tuple[list[str], str]:
    """若提供 csv_path 则直接读取；否则调用 run_whisper 并构建 CSV + frame_names。"""
    if csv_path is not None:
        text = csv_path.read_text(encoding="utf-8")
        # 从 CSV 反推出 frame_names
        reader = csv.DictReader(io.StringIO(text))
        frame_names_set = set()
        for row in reader:
            ff = row.get("frame_folders", "")
            if ff:
                for name in ff.split("|"):
                    n = name.strip()
                    if n:
                        frame_names_set.add(n)
        if not frame_names_set:
            raise ValueError("word_frames_csv 中未找到任何 frame_folders")
        frame_names = sorted(frame_names_set)
        return frame_names, text

    # 运行 Whisper
    cache_dir = whisper_cache
    if cache_dir is None:
        cache_dir = _REPO_ROOT / "Project" / "model" / "whisper_models"
    cache_dir = cache_dir.resolve()
    whisper_device = "cpu" if whisper_cpu or (not torch.cuda.is_available()) else "cuda"
    word_triples = run_whisper(
        audio_path,
        model_size=whisper_model,
        no_ssl_verify=no_ssl_verify,
        use_modelscope=use_modelscope,
        cache_dir=cache_dir,
        device=whisper_device,
    )
    return _build_word_frames_and_frames_from_audio(word_triples, fps)


def _load_checkpoint_and_model(checkpoint: Path, device: torch.device) -> tuple[AudioTextToMouth, dict]:
    """加载 checkpoint，返回模型与 word2id。"""
    ckpt = torch.load(checkpoint, map_location=device)
    word2id = ckpt.get("word2id")
    vocab_size = ckpt.get("vocab_size", len(word2id) if word2id is not None else None)
    if word2id is None or vocab_size is None:
        raise ValueError("checkpoint 中缺少 word2id/vocab_size，请使用 Project/train_audio_to_mouth.py 训练生成的 best.pt")

    state = dict(ckpt["model_state_dict"])
    # 兼容 transformers 新旧版本的 LayerNorm 权重命名
    for k in list(state.keys()):
        if "parametrizations.weight.original0" in k:
            new_k = k.replace("parametrizations.weight.original0", "weight_g")
            state[new_k] = state.pop(k)
        elif "parametrizations.weight.original1" in k:
            new_k = k.replace("parametrizations.weight.original1", "weight_v")
            state[new_k] = state.pop(k)

    use_text = ckpt.get("use_text", False)
    model = AudioTextToMouth(vocab_size=vocab_size, use_text=use_text).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, word2id


def run_infer_to_npz(
    checkpoint: Path,
    audio_path: Path,
    out_npz: Path,
    init_npz: Path | None,
    fps: float,
    word_frames_csv_path: Path | None,
    whisper_model: str,
    whisper_cache: Path | None,
    use_modelscope: bool,
    whisper_cpu: bool,
    no_ssl_verify: bool,
    device: torch.device,
    export_obj_dir: Path | None = None,
    mouth_scale: float = 1.0,
):
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    if init_npz is None:
        guess = audio_path.with_suffix(".npz")
        if guess.is_file():
            init_npz = guess
        else:
            raise FileNotFoundError("未指定 --init_npz，且未在音频目录找到同名 .npz")
    init_npz = Path(init_npz).resolve()
    if not init_npz.exists():
        raise FileNotFoundError(f"init_npz 不存在: {init_npz}")

    # 1) 载入 init（支持 .npz 文件或单帧目录，目录下需有 shape.npy, exp.npy, jawpose.npy, globalpose.npy）
    if init_npz.is_dir():
        shape = np.load(init_npz / "shape.npy").astype(np.float32)
        init_exp = np.load(init_npz / "exp.npy").astype(np.float32)
        init_jaw = np.load(init_npz / "jawpose.npy").astype(np.float32)
        init_global_pose = np.load(init_npz / "globalpose.npy").astype(np.float32)
        if init_exp.ndim > 1:
            init_exp = init_exp.squeeze()
        if init_jaw.ndim > 1:
            init_jaw = init_jaw.squeeze()
        if init_global_pose.ndim > 1:
            init_global_pose = init_global_pose.squeeze()
    else:
        init_data = dict(np.load(init_npz, allow_pickle=True))
        shape = np.asarray(init_data["shape"], dtype=np.float32)
        init_exp = np.asarray(init_data["init_exp"], dtype=np.float32)
        init_jaw = np.asarray(init_data["init_jaw"], dtype=np.float32)
        init_global_pose = np.asarray(init_data["init_global_pose"], dtype=np.float32)

    # 2) 获得 frame_names 与 word_frames_csv
    frame_names, csv_text = _load_or_build_word_frames_csv(
        audio_path=audio_path,
        fps=fps,
        whisper_model=whisper_model,
        whisper_cache=whisper_cache,
        use_modelscope=use_modelscope,
        whisper_cpu=whisper_cpu,
        no_ssl_verify=no_ssl_verify,
        csv_path=word_frames_csv_path,
    )
    num_frames = len(frame_names)

    # 3) 加载模型与词表
    model, word2id = _load_checkpoint_and_model(checkpoint, device)

    # 4) 为每个词片段跑一次模型，并累积到全局帧序列（每个词用「上一词最后一帧」作初始姿态）
    exp_pred = np.zeros((num_frames, N_EXP), dtype=np.float32)
    jaw_pred = np.zeros((num_frames, N_JAW), dtype=np.float32)
    global_pred = np.zeros((num_frames, N_GLOBAL_POSE), dtype=np.float32)
    counts = np.zeros((num_frames,), dtype=np.int32)

    # 当前词的初始姿态：第一个词用 npz 的 init，后续词用上一词预测的最后一帧
    cur_exp = np.array(init_exp, dtype=np.float32)
    cur_jaw = np.array(init_jaw, dtype=np.float32)
    cur_global_pose = np.array(init_global_pose, dtype=np.float32)

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    for row in rows:
        word = row.get("word", "").strip()
        try:
            start_sec = float(row.get("start_sec", "").strip() or 0)
            end_sec = float(row.get("end_sec", "").strip() or 0)
        except (ValueError, TypeError):
            continue
        if start_sec < 0 or end_sec <= start_sec:
            continue
        # 对应的帧索引区间（可能是小数需取整）
        try:
            start_frame = int(round(float(row.get("start_frame", 0))))
            end_frame = int(round(float(row.get("end_frame", 0))))
        except (ValueError, TypeError):
            continue
        frame_folders = row.get("frame_folders", "").strip().split("|") if row.get("frame_folders") else []
        frame_folders = [f.strip() for f in frame_folders if f.strip()]
        # 容错：用 frame_names 切片长度为 T
        if frame_folders:
            T = len(frame_folders)
            frame_indices = list(range(start_frame, start_frame + T))
        else:
            frame_indices = list(range(start_frame, end_frame + 1))
            T = len(frame_indices)
        if T <= 0:
            continue

        wave = load_audio_segment(audio_path, start_sec, end_sec, AUDIO_SAMPLE_RATE)
        word_id = word2id.get(word, word2id.get("<unk>", 0))

        batch = {
            "shape": torch.from_numpy(shape).unsqueeze(0).float().to(device),
            "exp": torch.from_numpy(cur_exp).unsqueeze(0).float().to(device),
            "jaw": torch.from_numpy(cur_jaw).unsqueeze(0).float().to(device),
            "global_pose": torch.from_numpy(cur_global_pose).unsqueeze(0).float().to(device),
            "raw_audio": torch.from_numpy(wave).unsqueeze(0).unsqueeze(0).float().to(device),
            "samplerate": [AUDIO_SAMPLE_RATE],
            "word_id": torch.tensor([word_id], dtype=torch.long, device=device),
            "mask": torch.ones(1, T, dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            pred = model(batch, desired_output_length=T)
        pred_np = pred[0].cpu().numpy()  # (T, 106)

        # 下一词的初始姿态 = 本词预测的最后一帧
        cur_exp = pred_np[T - 1, :N_EXP].copy()
        cur_jaw = pred_np[T - 1, N_EXP : N_EXP + N_JAW].copy()
        cur_global_pose = pred_np[T - 1, N_EXP + N_JAW :].copy()

        for local_t, f_idx in enumerate(frame_indices):
            if f_idx < 0 or f_idx >= num_frames:
                continue
            exp_pred[f_idx] += pred_np[local_t, :N_EXP]
            jaw_pred[f_idx] += pred_np[local_t, N_EXP : N_EXP + N_JAW]
            global_pred[f_idx] += pred_np[local_t, N_EXP + N_JAW :]
            counts[f_idx] += 1

    # 5) 对重叠帧取平均，未命中的帧用 init 值填充
    for i in range(num_frames):
        if counts[i] > 0:
            exp_pred[i] /= counts[i]
            jaw_pred[i] /= counts[i]
            global_pred[i] /= counts[i]
        else:
            exp_pred[i] = init_exp
            jaw_pred[i] = init_jaw
            global_pred[i] = init_global_pose

    # 口型幅度缩放：对相对初始姿态的偏移乘 scale（>1 放大张嘴幅度）
    if mouth_scale != 1.0:
        for i in range(num_frames):
            exp_pred[i] = init_exp + mouth_scale * (exp_pred[i] - init_exp)
            jaw_pred[i] = init_jaw + mouth_scale * (jaw_pred[i] - init_jaw)
            global_pred[i] = init_global_pose + mouth_scale * (global_pred[i] - init_global_pose)

    # 6) 按训练 npz 规范写出
    out_npz = out_npz.resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        shape=shape.astype(np.float32),
        exp=exp_pred,
        jaw=jaw_pred,
        global_pose=global_pred,
        frame_names=np.array(frame_names, dtype=object),
        init_shape=shape.astype(np.float32),
        init_exp=init_exp.astype(np.float32),
        init_jaw=init_jaw.astype(np.float32),
        init_global_pose=init_global_pose.astype(np.float32),
        word_frames_csv=np.array([csv_text], dtype=object),
    )
    print(f"Saved predicted npz to {out_npz} (N={num_frames} frames)")

    # 7) 可选导出 .obj 序列
    if export_obj_dir is not None:
        export_obj_dir = export_obj_dir.resolve()
        params_list = []
        for i in range(num_frames):
            params_list.append(
                {
                    "shape": shape,
                    "exp": exp_pred[i],
                    "jaw": jaw_pred[i],
                    "global_pose": global_pred[i],
                }
            )
        export_sequence_to_obj(params_list, export_obj_dir)
        print(f"Exported {num_frames} .obj to {export_obj_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audio + Whisper 对齐 → 预测口型 npz（与 Project/data 格式一致）")
    parser.add_argument("--checkpoint", type=Path, required=True, help="AudioTextToMouth checkpoint (best.pt)")
    parser.add_argument("--audio", type=Path, required=True, help="输入音频路径（.mp3）")
    parser.add_argument(
        "--out_npz",
        type=Path,
        default=None,
        help="输出 npz 路径（默认与音频同名，添加 _pred 后缀）",
    )
    parser.add_argument(
        "--init_npz",
        type=Path,
        default=None,
        help="初始姿态：.npz 文件，或单帧目录（含 shape.npy, exp.npy, jawpose.npy, globalpose.npy）；默认用 audio 同目录同名 .npz",
    )
    parser.add_argument(
        "--word_frames_csv",
        type=Path,
        default=None,
        help="可选：已有的 word_frames.csv（若提供则跳过 Whisper）",
    )
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help="词→帧对齐使用的帧率，默认与训练相同")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper/faster-whisper 模型规格或路径")
    parser.add_argument(
        "--whisper_cache",
        type=Path,
        default=None,
        help="Whisper 模型缓存目录，默认 Project/model/whisper_models",
    )
    parser.add_argument("--use_modelscope", action="store_true", help="从 ModelScope 下载 faster-whisper 模型")
    parser.add_argument("--whisper_cpu", action="store_true", help="强制 Whisper 在 CPU 上运行（避免 CUDA/cuDNN 问题）")
    parser.add_argument("--no_ssl_verify", action="store_true", help="下载模型时关闭 SSL 校验")
    parser.add_argument(
        "--export_obj_dir",
        type=Path,
        default=None,
        help="若提供，则基于预测 npz 导出 FLAME .obj 序列到该目录",
    )
    parser.add_argument(
        "--mouth_scale",
        type=float,
        default=1.0,
        help="口型幅度缩放：对相对初始姿态的偏移乘该系数，>1 放大张嘴幅度，默认 1.0",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    audio = args.audio
    if args.out_npz is None:
        stem = audio.with_suffix("").name
        args.out_npz = audio.with_name(f"{stem}_pred.npz")

    device = torch.device(args.device)
    run_infer_to_npz(
        checkpoint=args.checkpoint,
        audio_path=audio,
        out_npz=args.out_npz,
        init_npz=args.init_npz,
        fps=args.fps,
        word_frames_csv_path=args.word_frames_csv,
        whisper_model=args.whisper_model,
        whisper_cache=args.whisper_cache,
        use_modelscope=args.use_modelscope,
        whisper_cpu=args.whisper_cpu,
        no_ssl_verify=args.no_ssl_verify,
        device=device,
        export_obj_dir=args.export_obj_dir,
        mouth_scale=args.mouth_scale,
    )


if __name__ == "__main__":
    main()

