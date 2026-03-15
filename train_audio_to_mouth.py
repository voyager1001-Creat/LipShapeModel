"""
基于 Project/data 下打包的 npz + mp3 训练 AudioTextToMouth：
- 输入：shape/exp/jaw/global_pose 初始参数 + 音频片段 + 词 id
- 输出：口型 106 维序列 (exp 100 + jaw 3 + global_pose 3)

运行示例（在仓库根，确保已 pip install -e .）：
  python Project/train_audio_to_mouth.py --data_root Project/data --checkpoint_dir Project/checkpoints/audio_text_to_mouth
"""

import argparse
import csv
import io
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

# 仓库根（Project 的上一级）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Project.audio_to_mouth_config import (  # noqa: E402
    AUDIO_SAMPLE_RATE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LR,
    N_EXP,
    N_GLOBAL_POSE,
    N_JAW,
    OUTPUT_DIM,
    TRAIN_RATIO,
)
from Project.data import WordFrameDataset, collate_word_frames  # noqa: E402
from Project.model.audio_text_to_mouth import AudioTextToMouth  # noqa: E402


def _masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/gt (B,T,106), mask (B,T). 只对 mask==1 的位置求 MSE。"""
    diff = (pred - gt) ** 2
    masked = diff * mask.unsqueeze(-1)
    n = mask.sum() * pred.shape[-1]
    if n.item() == 0:
        return pred.new_zeros(())
    return masked.sum() / n.clamp(min=1)


def _scan_npz_and_build_vocab(
    data_root: Path,
    ids: list[str] | None = None,
) -> tuple[list[Path], list[Path], dict, list[int]]:
    """
    扫描 data_root 下的 <id>/<id>.npz + <id>.mp3，统计：
    - npz_paths: 每个 id 的 npz 路径
    - audio_paths: 每个 id 的 mp3 路径
    - word2id: 全局词表（含 <unk>=0, <pad>=1）
    - n_rows_list: 每个 npz 内 word_frames 行数
    """
    data_root = Path(data_root)
    if ids is None:
        # 目录名视为 id：要求 <id>/<id>.npz 存在
        candidates = sorted(p for p in data_root.iterdir() if p.is_dir())
    else:
        candidates = [data_root / str(i) for i in ids]

    npz_paths: list[Path] = []
    audio_paths: list[Path] = []
    n_rows_list: list[int] = []
    word2id: dict[str, int] = {"<unk>": 0, "<pad>": 1}

    for d in candidates:
        stem = d.name
        npz_path = d / f"{stem}.npz"
        audio_path = d / f"{stem}.mp3"
        if not npz_path.is_file() or not audio_path.is_file():
            continue
        data = dict(np.load(npz_path, allow_pickle=True))
        csv_text = data.get("word_frames_csv")
        if csv_text is None:
            continue
        if hasattr(csv_text, "item"):
            csv_text = csv_text.item()
        # 统计行数 + 更新词表
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        if not rows:
            continue
        n_rows = len(rows)
        for row in rows:
            w = row.get("word", "").strip()
            if w and w not in word2id:
                word2id[w] = len(word2id)
        npz_paths.append(npz_path)
        audio_paths.append(audio_path)
        n_rows_list.append(n_rows)

    if not npz_paths:
        raise FileNotFoundError(f"在 {data_root} 下未找到任何 <id>/<id>.npz + <id>.mp3 组合")
    return npz_paths, audio_paths, word2id, n_rows_list


def _build_datasets(
    npz_paths: list[Path],
    audio_paths: list[Path],
    word2id: dict,
) -> ConcatDataset:
    """为每个 npz 建立一个 WordFrameDataset，然后用 ConcatDataset 拼接。"""
    datasets = []
    for npz_path, audio_path in zip(npz_paths, audio_paths):
        ds = WordFrameDataset(
            audio_path=audio_path,
            word2id=word2id,
            packed_npz_path=npz_path,
        )
        datasets.append(ds)
    return ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser(description="Train AudioTextToMouth on Project/data/<id>/<id>.npz + <id>.mp3")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("Project") / "data",
        help="包含 <id>/<id>.npz 和 <id>.mp3 的根目录",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="逗号分隔的子目录 id，仅使用这些 id（如 1,2,3）；留空则使用 data_root 下全部子目录",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="保存 checkpoint 的目录，默认 Project/checkpoints/audio_text_to_mouth",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = args.data_root
    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or None

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) 扫描数据与词表
    npz_paths, audio_paths, word2id, n_rows_list = _scan_npz_and_build_vocab(data_root, ids)
    print(f"Found {len(npz_paths)} sequences, vocab_size={len(word2id)}")

    # 2) 构建 ConcatDataset（内部会过滤掉 start_sec/end_sec 为空的无效行）
    concat_ds = _build_datasets(npz_paths, audio_paths, word2id)
    total_rows = len(concat_ds)
    print(f"Valid word-level samples after filtering: {total_rows}")
    if total_rows == 0:
        raise ValueError("过滤后没有有效样本，请检查 npz 内 word_frames_csv 的 start_sec/end_sec 是否为空")

    # 3) 生成全局 train/val 索引
    indices = list(range(total_rows))
    random.shuffle(indices)
    n_train = int(total_rows * args.train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    train_ds = Subset(concat_ds, train_indices)
    val_ds = Subset(concat_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_word_frames,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_word_frames,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = AudioTextToMouth(vocab_size=len(word2id), use_text=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            pred = model(batch)
            loss = _masked_mse(pred, batch["gt"], batch["mask"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                pred = model(batch)
                loss = _masked_mse(pred, batch["gt"], batch["mask"])
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = args.checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "word2id": word2id,
                    "vocab_size": len(word2id),
                    "use_text": getattr(model, "use_text", False),
                    "audio_sample_rate": AUDIO_SAMPLE_RATE,
                    "output_dim": OUTPUT_DIM,
                    "n_exp": N_EXP,
                    "n_jaw": N_JAW,
                    "n_global_pose": N_GLOBAL_POSE,
                },
                ckpt,
            )
            print(f"  -> saved {ckpt}")

    print("Done. Best val_loss =", best_val)


if __name__ == "__main__":
    main()

