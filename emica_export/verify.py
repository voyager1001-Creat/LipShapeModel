"""
验证 emica_export 复用是否成功：不依赖 inferno，能加载 FLAME 并导出一帧 .obj。
运行方式（仓库根目录）：
  python -m Project.emica_export.verify
或（在 Project 目录下）：
  python emica_export/verify.py
"""
from pathlib import Path
import sys

# 保证可从 Project 或仓库根运行
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR.parent))  # 仓库根，以便 import Project.emica_export


def main():
    print("1. 检查是否不依赖 inferno ...")
    try:
        from Project.emica_export import (
            load_flame,
            params_to_verts,
            export_106_to_obj,
            load_params_from_dir,
        )
        print("   OK — 已从 Project.emica_export 导入，未使用 inferno。")
    except ImportError as e:
        # 若在 Project 下运行，可能没有 Project 包名
        try:
            from emica_export import (
                load_flame,
                params_to_verts,
                export_106_to_obj,
                load_params_from_dir,
            )
            print("   OK — 已从 emica_export 导入（当前在 Project 下）。")
        except ImportError:
            print("   失败:", e)
            print("   请从仓库根执行: python -m Project.emica_export.verify")
            return 1

    model_root = _PROJECT_DIR / "model"
    geo = model_root / "FLAME" / "geometry"
    required = [
        "generic_model.pkl",
        "landmark_embedding.npy",
        "mediapipe_landmark_embedding.npz",
    ]
    print("2. 检查 FLAME 几何路径:", geo)
    missing = [f for f in required if not (geo / f).is_file()]
    if missing:
        print("   缺少文件:", missing)
        print("   请从 inferno 的 assets/FLAME/geometry/ 拷贝到上述目录。")
        return 1
    print("   OK — 所需文件均在。")

    print("3. 加载 FLAME 并导出一帧测试 .obj ...")
    import numpy as np
    import torch

    # 清空 FLAME 缓存以便验证完整加载
    try:
        import Project.emica_export.params_to_obj as _m
    except ImportError:
        import emica_export.params_to_obj as _m
    _m._flame_cache = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame = load_flame(model_root=str(model_root), device=device)
    dummy_params = {
        "shape": np.zeros(300, dtype=np.float32),
        "exp": np.zeros(100, dtype=np.float32),
        "jaw": np.zeros(3, dtype=np.float32),
        "global_pose": np.zeros(3, dtype=np.float32),
    }
    out_path = _PROJECT_DIR / "test_export_frame.obj"
    export_106_to_obj(
        dummy_params,
        out_path,
        flame=flame,
        model_root=str(model_root),
        device=device,
    )
    if out_path.is_file():
        print("   OK — 已写出:", out_path)
    else:
        print("   失败 — 未生成文件:", out_path)
        return 1

    print("")
    print("复用验证通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
