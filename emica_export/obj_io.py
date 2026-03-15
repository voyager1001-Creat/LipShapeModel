# Minimal OBJ writer for vertices + faces. No texture/uv. Standalone, no inferno dependency.

import numpy as np
from pathlib import Path


def write_obj(filepath, vertices, faces):
    """
    Save mesh as .obj (vertices and faces only).

    Args:
        filepath: str or Path, output .obj path.
        vertices: array (nver, 3), vertex positions.
        faces: array (ntri, 3), 0-based triangle indices.
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() != ".obj":
        filepath = filepath.with_suffix(".obj")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).copy()
    # OBJ uses 1-based indices
    faces += 1
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for i in range(vertices.shape[0]):
            f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        for i in range(faces.shape[0]):
            f.write("f {} {} {}\n".format(faces[i, 0], faces[i, 1], faces[i, 2]))
