# Copied from inferno/models/DecaFLAME.py for standalone Project. Uses .lbs only.
# FLAME and FLAME_mediapipe: 106-dim params -> vertices. No inferno dependency.

import torch
import torch.nn as nn
import numpy as np
import pickle

from . import lbs


def _patch_numpy_for_chumpy():
    """numpy 1.24+ / 2.0 移除了 np.int/float/bool/complex 等，chumpy 依赖它们。用 float64/int64 等补回。"""
    if getattr(np, "int", None) is not None and getattr(np.int, "view", None) is not None:
        return
    np.int = np.int64
    np.float = np.float64
    np.bool = getattr(np, "bool_", np.dtype("bool"))
    np.complex = getattr(np, "complex_", np.dtype("complex128"))
    np.object = getattr(np, "object_", object)
    np.unicode = getattr(np, "unicode_", str)
    np.str = getattr(np, "str_", str)


def _load_flame_pkl(path):
    """加载 FLAME generic_model.pkl（依赖 chumpy，见 Project/requirements.txt）。"""
    _patch_numpy_for_chumpy()
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
    return array


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(
        rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
    )
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def _cfg(config, key, default=None):
    """Config 支持 SimpleNamespace 或 dict。"""
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


class FLAME(nn.Module):
    """FLAME decoder: shape_params(300), expression_params(100), pose_params(6) -> vertices."""

    def __init__(self, config):
        super(FLAME, self).__init__()
        flame_model_path = str(_cfg(config, "flame_model_path") or "")
        ss = _load_flame_pkl(flame_model_path)
        flame_model = Struct(**ss)

        self.cfg = config
        self.dtype = torch.float32
        n_shape = _cfg(config, "n_shape", 300)
        n_exp = _cfg(config, "n_exp", 100)

        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, :n_shape],
                shapedirs[:, :, 300 : 300 + n_exp],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        default_eyeball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter("eye_pose", nn.Parameter(default_eyeball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter("neck_pose", nn.Parameter(default_neck_pose, requires_grad=False))

        lmk_path = str(_cfg(config, "flame_lmk_embedding_path") or "")
        lmk_embeddings = np.load(lmk_path, allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.tensor(lmk_embeddings["static_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx",
            torch.tensor(lmk_embeddings["dynamic_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            torch.tensor(lmk_embeddings["dynamic_lmk_bary_coords"], dtype=self.dtype),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        NECK_IDX = 1
        neck_kin_chain = []
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32
    ):
        batch_size = pose.shape[0]
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = lbs.batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(
            batch_size, -1, 3, 3
        )
        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
    ):
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            n_exp = getattr(self.cfg, "n_exp", 100)
            expression_params = torch.zeros(batch_size, n_exp).to(shape_params.device)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs.lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            detach_pose_correctives=False,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)
        landmarks2d = lbs.vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = lbs.vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )
        return vertices, landmarks2d, landmarks3d


class FLAME_mediapipe(FLAME):
    """FLAME with MediaPipe landmark output; same forward for vertices."""

    def __init__(self, config):
        super().__init__(config)
        mp_path = str(_cfg(config, "flame_mediapipe_lmk_embedding_path") or "")
        lmk_mp = np.load(mp_path, allow_pickle=True, encoding="latin1")
        self.register_buffer(
            "lmk_faces_idx_mediapipe",
            torch.tensor(lmk_mp["lmk_face_idx"].astype(np.int64), dtype=torch.long),
        )
        self.register_buffer(
            "lmk_bary_coords_mediapipe",
            torch.tensor(lmk_mp["lmk_b_coords"], dtype=self.dtype),
        )

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        eye_pose_params=None,
    ):
        vertices, landmarks2d, landmarks3d = super().forward(
            shape_params, expression_params, pose_params, eye_pose_params
        )
        batch_size = shape_params.shape[0]
        lmk_faces_idx_mp = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(
            batch_size, -1
        ).contiguous()
        lmk_bary_coords_mp = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        ).contiguous()
        landmarks2d_mediapipe = lbs.vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx_mp, lmk_bary_coords_mp
        )
        return vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe
