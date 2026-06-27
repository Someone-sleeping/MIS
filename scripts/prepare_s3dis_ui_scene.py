import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from interactive_tool.utils import write_ply


S3DIS_CLASS_IDS = {
    "ceiling": 1,
    "floor": 2,
    "wall": 3,
    "beam": 4,
    "column": 5,
    "window": 6,
    "door": 7,
    "table": 8,
    "chair": 9,
    "sofa": 10,
    "bookcase": 11,
    "board": 12,
    "clutter": 13,
}


def _read_annotation_file(path):
    try:
        data = np.loadtxt(path, dtype=np.float32)
    except ValueError:
        data = np.loadtxt(path, dtype=np.float32, usecols=(0, 1, 2, 3, 4, 5))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :6]


def _room_to_arrays(room_dir, max_points, exclude_classes=None):
    exclude_classes = set(exclude_classes or [])
    annotation_dir = room_dir / "Annotations"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing S3DIS Annotations directory: {annotation_dir}")

    coords, colors, semantic_labels, instance_labels = [], [], [], []
    instance_id = 1
    for annotation_path in sorted(annotation_dir.glob("*.txt")):
        class_name = annotation_path.stem.split("_")[0]
        if class_name in exclude_classes:
            continue
        class_id = S3DIS_CLASS_IDS.get(class_name, S3DIS_CLASS_IDS["clutter"])
        data = _read_annotation_file(annotation_path)
        if data.size == 0:
            continue
        count = data.shape[0]
        coords.append(data[:, :3])
        colors.append(np.clip(data[:, 3:6], 0, 255))
        semantic_labels.append(np.full(count, class_id, dtype=np.int32))
        instance_labels.append(np.full(count, instance_id, dtype=np.int32))
        instance_id += 1

    if not coords:
        raise ValueError(f"No annotation points found in {annotation_dir}")

    coords = np.concatenate(coords, axis=0).astype(np.float32)
    colors = np.concatenate(colors, axis=0).astype(np.uint8)
    semantic_labels = np.concatenate(semantic_labels, axis=0).astype(np.int32)
    instance_labels = np.concatenate(instance_labels, axis=0).astype(np.int32)

    if max_points and coords.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(coords.shape[0], size=max_points, replace=False))
        coords = coords[keep]
        colors = colors[keep]
        semantic_labels = semantic_labels[keep]
        instance_labels = instance_labels[keep]

    coords = coords - coords.min(axis=0, keepdims=True)
    distance = np.linalg.norm(coords, axis=1).astype(np.float32)
    intensity = (colors.astype(np.float32).mean(axis=1) / 255.0).astype(np.float32)
    time = np.zeros(coords.shape[0], dtype=np.float32)
    return coords, colors, time, intensity, distance, semantic_labels, instance_labels


def main():
    parser = argparse.ArgumentParser(description="Export one S3DIS room as an Interactive4D GUI scene.")
    parser.add_argument("--s3dis_root", default="data/S3DIS/Stanford3dDataset_v1.2")
    parser.add_argument("--area", default="Area_5")
    parser.add_argument("--room", default=None, help="Room name such as office_1. Defaults to the first room in the area.")
    parser.add_argument("--output_dir", default="interactive_scenes_s3dis")
    parser.add_argument("--max_points", type=int, default=60000, help="0 keeps all room points")
    parser.add_argument("--exclude_classes", nargs="*", default=[], help="S3DIS classes to remove, e.g. ceiling")
    args = parser.parse_args()

    s3dis_root = Path(args.s3dis_root)
    area_dir = s3dis_root / args.area
    if args.room is None:
        rooms = sorted(path.name for path in area_dir.iterdir() if (path / "Annotations").is_dir())
        if not rooms:
            raise FileNotFoundError(f"No S3DIS rooms found under {area_dir}")
        room = rooms[0]
    else:
        room = args.room

    room_dir = area_dir / room
    coords, colors, time, intensity, distance, semantic_labels, instance_labels = _room_to_arrays(
        room_dir,
        args.max_points,
        exclude_classes=args.exclude_classes,
    )

    scene_name = f"scene_{args.area}_{room}"
    scene_dir = Path(args.output_dir) / scene_name
    os.makedirs(scene_dir, exist_ok=True)
    output_path = scene_dir / "scan.ply"
    write_ply(
        str(output_path),
        [coords, colors, time, intensity, distance, semantic_labels, instance_labels],
        ["x", "y", "z", "red", "green", "blue", "time", "intensity", "distance", "semantic_label", "label"],
    )
    excluded = ", ".join(args.exclude_classes) if args.exclude_classes else "none"
    print(f"Wrote {output_path} with {coords.shape[0]} points from {room_dir}; excluded classes: {excluded}")


if __name__ == "__main__":
    main()
