# Standalone 106-dim params -> FLAME vertices -> .obj. No inferno dependency.

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from .flame import FLAME_mediapipe
from .obj_io import write_obj

# Default model root: Project/model
_DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
_flame_cache = None


def _flame_config(model_root: Path) -> SimpleNamespace:
    """Build FLAME config with paths under model_root/FLAME/geometry/."""
    model_root = Path(model_root)
    geo = model_root / "FLAME" / "geometry"
    return SimpleNamespace(
        flame_model_path=str(geo / "generic_model.pkl"),
        flame_lmk_embedding_path=str(geo / "landmark_embedding.npy"),
        flame_mediapipe_lmk_embedding_path=str(geo / "mediapipe_landmark_embedding.npz"),
        n_shape=300,
        n_exp=100,
    )


def load_flame(model_root=None, device=None):
    """
    Load FLAME_mediapipe from Project/model/FLAME/geometry/ (or model_root).
    Returns FLAME module; use .faces_tensor for writing .obj.
    Cached per process so repeated calls with same model_root return same instance.
    """
    global _flame_cache
    if model_root is None:
        model_root = _DEFAULT_MODEL_ROOT
    model_root = Path(model_root)
    if _flame_cache is not None:
        return _flame_cache
    cfg = _flame_config(model_root)
    flame = FLAME_mediapipe(cfg)
    if device is not None:
        flame = flame.to(device)
    flame.eval()
    _flame_cache = flame
    return flame


def params_to_verts(flame, params: dict, device: torch.device) -> np.ndarray:
    """
    Decode shape/exp/jaw/global_pose to vertices (single frame).
    params: dict with keys shape (300,), exp (100,), jaw (3,), global_pose (3,).
            Supports aliases: jawpose -> jaw, globalpose -> global_pose.
    Returns: (V, 3) numpy array.
    """
    shape = torch.from_numpy(np.asarray(params["shape"], dtype=np.float32)).float().to(device)
    exp = torch.from_numpy(np.asarray(params["exp"], dtype=np.float32)).float().to(device)
    global_pose = torch.from_numpy(
        np.asarray(params.get("global_pose", params.get("globalpose")), dtype=np.float32)
    ).float().to(device)
    jaw = torch.from_numpy(
        np.asarray(params.get("jaw", params.get("jawpose", np.zeros(3))), dtype=np.float32)
    ).float().to(device)
    if shape.ndim == 1:
        shape = shape.unsqueeze(0)
        exp = exp.unsqueeze(0)
        global_pose = global_pose.unsqueeze(0)
        jaw = jaw.unsqueeze(0)
    pose_params = torch.cat([global_pose, jaw], dim=-1)
    with torch.no_grad():
        out = flame(
            shape_params=shape,
            expression_params=exp,
            pose_params=pose_params,
            eye_pose_params=None,
        )
    verts = out[0].cpu().numpy().squeeze()
    return verts


def load_params_from_dir(params_dir: Path, frame: int = 0) -> dict:
    """
    Load shape, exp, jaw, global_pose from a directory.
    Supports: params.npz; or frame_XXXXX/ with .npy files; or single dir with shape.npy, exp.npy, jawpose.npy, globalpose.npy.
    """
    params_dir = Path(params_dir)
    npz_file = params_dir / "params.npz"
    if npz_file.is_file():
        data = dict(np.load(npz_file, allow_pickle=False))
        for k in list(data.keys()):
            arr = data[k]
            if arr.ndim >= 2 and arr.shape[1] > 1:
                data[k] = (arr[0, frame] if arr.shape[0] == 1 else arr[frame]).squeeze()
            else:
                data[k] = arr.squeeze()
        return data
    frame_dir = params_dir / f"frame_{frame:05d}"
    if frame_dir.is_dir():
        out = {}
        for key, fname in [
            ("shape", "shape"),
            ("exp", "exp"),
            ("jaw", "jaw"),
            ("jaw", "jawpose"),
            ("global_pose", "global_pose"),
            ("global_pose", "globalpose"),
        ]:
            if key in out:
                continue
            f = frame_dir / f"{fname}.npy"
            if f.is_file():
                out[key] = np.load(f)
        return out if out else None
    name_map = {
        "shape": "shape",
        "exp": "exp",
        "jaw": "jaw",
        "jawpose": "jaw",
        "global_pose": "global_pose",
        "globalpose": "global_pose",
    }
    out = {}
    for key, file_name in name_map.items():
        if key in out:
            continue
        f = params_dir / f"{file_name}.npy"
        if f.is_file():
            out[key] = np.load(f)
    return out if out else None


def export_106_to_obj(
    params,
    out_path,
    flame=None,
    model_root=None,
    device=None,
):
    """
    Export one frame of 106-dim params to a single .obj file.
    params: dict (shape, exp, jaw, global_pose) or Path to a directory (load_params_from_dir).
    out_path: path for the output .obj file.
    flame: if None, load_flame(model_root, device) is used (cached).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(params, (Path, str)):
        params = load_params_from_dir(Path(params), frame=0)
    if not params or "shape" not in params:
        raise ValueError("params must contain at least 'shape' and 'exp' (and jaw, global_pose)")
    if flame is None:
        flame = load_flame(model_root=model_root, device=device)
    verts = params_to_verts(flame, params, device)
    faces = flame.faces_tensor.cpu().numpy()
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)
    elif faces.ndim == 3:
        faces = faces[0]
    write_obj(out_path, verts, faces)


def export_sequence_to_obj(
    frame_dirs_or_params_list,
    out_dir,
    flame=None,
    model_root=None,
    device=None,
    name_pattern="mesh_frame_{:05d}.obj",
):
    """
    Export multiple frames to .obj files in out_dir.
    frame_dirs_or_params_list: list of Path (frame dirs) or list of dict (params).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flame is None:
        flame = load_flame(model_root=model_root, device=device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(frame_dirs_or_params_list):
        if isinstance(item, (Path, str)):
            params = load_params_from_dir(Path(item), frame=0)
        else:
            params = item
        out_path = out_dir / name_pattern.format(i)
        export_106_to_obj(params, out_path, flame=flame, device=device)
