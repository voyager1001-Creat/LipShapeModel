"""Quick npz inspection: keys, shapes, first N items."""
import sys
from pathlib import Path
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\GitHub\famle\inferno\Project\data\2\2.npz"
    path = Path(path)
    if not path.exists():
        print("File not found:", path)
        return
    print("Path:", path.resolve())
    print("File size (MB):", round(path.stat().st_size / (1024 * 1024), 4))
    z = np.load(path, allow_pickle=True)
    print("Keys:", list(z.keys()))
    for k in z.files:
        x = z[k]
        if hasattr(x, "shape"):
            print(f"  {k}: shape={x.shape}, dtype={x.dtype}")
        else:
            print(f"  {k}: {type(x).__name__}")
    if "exp" in z.files:
        print("exp first 3 rows (first 5 dims):", z["exp"][:3, :5])
    if "frame_names" in z.files:
        print("frame_names first 10:", list(z["frame_names"][:10]))
    if "word_frames_csv" in z.files:
        csv = str(z["word_frames_csv"][()])
        lines = csv.strip().split("\n")
        print("word_frames_csv lines:", len(lines), "| first 3 lines:", lines[:3])
    z.close()

if __name__ == "__main__":
    main()
