"""
视频 → 训练数据一条龙：输入视频（+ 可选单独音频），输出 out_dir/<视频名>.npz 与 out_dir/<视频名>.mp3。
例如 --video Project/data/2/2.mp4 --out_dir Project/data/2 则得到 2.npz 和 2.mp3，不复制、直接使用同目录已有同名 .mp3。
不写大量中间帧目录，仅在临时目录解帧与人脸检测，完成后删除。

运行：在仓库根执行。若报 No module named 'inferno.datasets'，先在仓库根执行：
  pip install -e .
再运行：
  python Project/scripts/video_to_training_data.py --video Project/data/2/2.mp4 --out_dir Project/data/2
"""
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import auto

# 依赖仓库根在 PYTHONPATH
from inferno.datasets.FaceVideoDataModule import TestFaceVideoDM
from inferno_apps.FaceReconstruction.utils.load import load_model
from inferno_apps.FaceReconstruction.utils.test import test
from inferno.utils.other import get_path_to_assets


def _parse_fps(fps_str):
    """将 video_metas 的 fps（如 '30/1'）解析为 float。"""
    if fps_str is None:
        return None
    s = str(fps_str).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _ensure_audio_to_out_dir(video_path: Path, audio_path: Optional[Path], out_dir: Path, stem: str) -> Path:
    """确保 out_dir/<stem>.mp3 可用：若提供 audio_path 则复制到 out_dir；否则若视频同目录已有 <stem>.mp3 且 out_dir 即该目录则直接使用，否则复制；否则用 ffmpeg 从视频抽取到 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp3 = out_dir / (stem + ".mp3")
    if audio_path is not None and Path(audio_path).is_file():
        if Path(audio_path).resolve() != out_mp3.resolve():
            shutil.copy2(audio_path, out_mp3)
        return out_mp3
    # 未指定 --audio：在视频所在目录下查找与视频同名的 .mp3
    same_name_mp3 = video_path.parent / (stem + ".mp3")
    if same_name_mp3.is_file():
        if same_name_mp3.resolve() == out_mp3.resolve():
            return out_mp3  # 已是同一文件，直接使用
        shutil.copy2(same_name_mp3, out_mp3)
        return out_mp3
    # 从视频抽取
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
        str(out_mp3),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp3


def _build_word_frames_csv(word_triples, fps: float, num_frames: int, frame_names: list) -> str:
    """根据 Whisper 词时间戳和 fps 拼出 word_frames CSV 字符串（表头 + 行，frame_folders 用 | 连接）。"""
    lines = ["word,start_sec,end_sec,start_frame,end_frame,frame_folders"]
    for word, start_sec, end_sec in word_triples:
        start_frame = max(0, min(round(start_sec * fps), num_frames - 1))
        end_frame = max(0, min(round(end_sec * fps), num_frames - 1))
        if start_frame > end_frame:
            end_frame = start_frame
        frame_folders = frame_names[start_frame : end_frame + 1]
        lines.append(
            f"{word},{round(start_sec, 4)},{round(end_sec, 4)},{start_frame},{end_frame},"
            f"{'|'.join(frame_folders)}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Video → <视频名>.npz + <视频名>.mp3 训练数据（FaceReconstruction + Whisper）")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--audio", type=Path, default=None,
                        help="可选单独音频；省略时先查视频所在目录下与视频同名的 .mp3，无则从视频抽取")
    parser.add_argument("--out_dir", type=Path, required=True, help="输出目录，将写入 out_dir/<视频名>.npz 与 out_dir/<视频名>.mp3")
    parser.add_argument("--path_to_models", type=str, default=None,
                        help="FaceReconstruction 模型目录，默认 get_path_to_assets()/FaceReconstruction/models")
    parser.add_argument("--model_name", type=str, default="EMICA-CVT_flame2020_notexture",
                        help="模型名，与 demo_face_rec_on_video 一致")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="词→帧对齐用帧率；若能从视频元数据解析则覆盖")
    parser.add_argument("--use_cpu", action="store_true", help="全部使用 CPU（否则优先 GPU）")
    parser.add_argument("--whisper_cpu", action="store_true",
                        help="仅 Whisper 使用 CPU（人脸重建仍用 GPU；遇 cuDNN/libcudnn_ops 报错时用此选项）")
    parser.add_argument("--use_modelscope", action="store_true", help="Whisper 从 ModelScope 下载")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper 模型：base/small/medium 或本地路径")
    parser.add_argument("--whisper_cache", type=Path, default=None,
                        help="Whisper 模型/缓存目录，默认 Project/model/whisper_models")
    parser.add_argument("--no_ssl_verify", action="store_true", help="Whisper 下载时关闭 SSL 校验")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    stem = video_path.stem  # 输出文件名与视频同名，如 2.mp4 → 2.npz / 2.mp3
    out_dir = Path(args.out_dir).resolve()
    temp_dir = out_dir / "_temp"
    path_to_models = args.path_to_models
    if path_to_models is None:
        path_to_models = str(Path(get_path_to_assets()) / "FaceReconstruction" / "models")

    device = torch.device("cpu" if args.use_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    try:
        # 1) DM：解帧 + 人脸检测到临时目录
        print("Preparing video (extract frames + face detection)...")
        dm = TestFaceVideoDM(
            video_path,
            str(temp_dir),
            processed_subfolder=None,
            batch_size=4,
            num_workers=4,
            face_detector="fan3d",
            device=device,
        )
        dm.prepare_data()
        dm.setup()

        # 2) FPS：从 video_metas 解析，失败用 --fps
        fps = args.fps
        if getattr(dm, "video_metas", None) and len(dm.video_metas) > 0 and dm.video_metas[0]:
            parsed = _parse_fps(dm.video_metas[0].get("fps"))
            if parsed is not None:
                fps = parsed
        print(f"FPS: {fps}")

        # 3) 加载模型并遍历 test_dataloader，在内存中收集 codes（不调用 save_codes）
        print("Loading FaceReconstruction model...")
        model, _ = load_model(path_to_models, args.model_name)
        model.to(device)
        model.eval()

        frame_names: list = []
        exp_list: list = []
        jaw_list: list = []
        global_pose_list: list = []
        shape_first = None

        dl = dm.test_dataloader()
        for batch in auto.tqdm(dl, desc="Reconstructing"):
            vals = test(model, batch)
            bs = batch["image"].shape[0]
            names = batch["image_name"]
            for i in range(bs):
                name = names[i] if isinstance(names[i], str) else names[i]
                frame_names.append(name)
                e = vals["expcode"][i].detach().cpu().numpy()
                j = vals["jawpose"][i].detach().cpu().numpy()
                g = vals["globalpose"][i].detach().cpu().numpy()
                exp_list.append(e)
                jaw_list.append(j)
                global_pose_list.append(g)
                if shape_first is None:
                    shape_first = vals["shapecode"][i].detach().cpu().numpy()

        if shape_first is None or not frame_names:
            raise RuntimeError("未得到任何有效帧")

        exp = np.stack(exp_list).astype(np.float32)
        jaw = np.stack(jaw_list).astype(np.float32)
        global_pose = np.stack(global_pose_list).astype(np.float32)
        shape = shape_first.astype(np.float32)
        num_frames = len(frame_names)

        # 4) 确保 out_dir/<stem>.mp3 可用（直接使用同目录已有 / 复制 / 抽取）
        print("Preparing audio...")
        audio_path = _ensure_audio_to_out_dir(video_path, args.audio, out_dir, stem)

        # 5) Whisper 词对齐
        print("Running Whisper...")
        try:
            from demo.align_audio_words_to_frames import run_whisper
        except ImportError:
            from align_audio_words_to_frames import run_whisper  # 兼容直接跑 demo 目录
        cache_dir = Path(args.whisper_cache).resolve() if args.whisper_cache else None
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "Project" / "model" / "whisper_models"
        whisper_device = "cpu" if (args.use_cpu or args.whisper_cpu) else ("cuda" if torch.cuda.is_available() else "cpu")
        word_triples = run_whisper(
            audio_path,
            model_size=args.whisper_model,
            no_ssl_verify=args.no_ssl_verify,
            use_modelscope=args.use_modelscope,
            cache_dir=cache_dir,
            device=whisper_device,
        )

        # 6) 拼 word_frames CSV
        word_frames_csv = _build_word_frames_csv(word_triples, fps, num_frames, frame_names)

        # 7) 写 <stem>.npz（与 pack_result_to_npz 结构一致）
        init_exp = exp[0:1].squeeze(0)
        init_jaw = jaw[0:1].squeeze(0)
        init_global_pose = global_pose[0:1].squeeze(0)

        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / (stem + ".npz")
        np.savez_compressed(
            npz_path,
            shape=shape,
            exp=exp,
            jaw=jaw,
            global_pose=global_pose,
            frame_names=np.array(frame_names, dtype=object),
            init_shape=shape,
            init_exp=init_exp,
            init_jaw=init_jaw,
            init_global_pose=init_global_pose,
            word_frames_csv=np.array([word_frames_csv], dtype=object),
        )
        print(f"Saved {npz_path} (N={num_frames} frames), {out_dir / (stem + '.mp3')}")

    finally:
        # 8) 删除临时目录
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Removed temp dir:", temp_dir)


if __name__ == "__main__":
    main()
