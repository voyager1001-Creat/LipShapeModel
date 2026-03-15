"""
Standalone 106-dim (shape, exp, jaw, global_pose) -> FLAME mesh -> .obj.
No dependency on inferno repo. Requires only torch, numpy, and FLAME geometry files.

Place under Project/model/FLAME/geometry/:
  - generic_model.pkl
  - landmark_embedding.npy
  - mediapipe_landmark_embedding.npz
(Copy from inferno assets/FLAME/geometry/ or FLAME official data.)
"""

from .params_to_obj import (
    load_flame,
    params_to_verts,
    load_params_from_dir,
    export_106_to_obj,
    export_sequence_to_obj,
)
from .obj_io import write_obj
from .flame import FLAME, FLAME_mediapipe

__all__ = [
    "load_flame",
    "params_to_verts",
    "load_params_from_dir",
    "export_106_to_obj",
    "export_sequence_to_obj",
    "write_obj",
    "FLAME",
    "FLAME_mediapipe",
]
