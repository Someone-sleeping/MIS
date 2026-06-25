import os
from pathlib import Path
from random import random, shuffle
from typing import Optional

import numpy as np
import volumentations as V
from torch.utils.data import Dataset


S3DIS_LABELS = {
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


class S3DISDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = "data/S3DIS/Stanford3dDataset_v1.2",
        mode: Optional[str] = "train",
        volume_augmentations_path: Optional[str] = None,
        sweep: Optional[int] = 1,
        center_coordinates=True,
        window_overlap=0,
        validation_area: str = "Area_5",
        max_points_per_room: Optional[int] = 200000,
    ):
        super(S3DISDataset, self).__init__()

        if sweep != 1:
            raise ValueError("S3DIS is a static indoor dataset and currently supports only sweep=1.")

        self.mode = mode
        self.data_dir = Path(data_dir)
        self.center_coordinates = center_coordinates
        self.validation_area = validation_area
        self.max_points_per_room = max_points_per_room

        if not self.data_dir.exists():
            raise FileNotFoundError(f"S3DIS raw data directory not found at {self.data_dir}")

        self.data = self._discover_rooms()
        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No S3DIS room annotations found under {self.data_dir}. "
                "Expected Area_x/<room>/Annotations/*.txt."
            )

        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")

    def _discover_rooms(self):
        rooms = []
        for area_dir in sorted(self.data_dir.glob("Area_*")):
            if not area_dir.is_dir():
                continue
            is_validation = area_dir.name == self.validation_area
            if ("train" in self.mode and is_validation) or ("validation" in self.mode and not is_validation):
                continue
            for room_dir in sorted(area_dir.iterdir()):
                ann_dir = room_dir / "Annotations"
                if not ann_dir.is_dir():
                    continue
                annotation_files = sorted(ann_dir.glob("*.txt"))
                if annotation_files:
                    rooms.append(
                        {
                            "area": area_dir.name,
                            "room": room_dir.name,
                            "annotation_files": annotation_files,
                        }
                    )
        return rooms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        room = self.data[idx]

        coordinates_list = []
        rgb_list = []
        labels_list = []
        obj2label_map = {}
        click_idx = {"0": []}
        obj_id = 1

        for annotation_file in room["annotation_files"]:
            class_name = annotation_file.stem.split("_")[0].lower()
            semantic_id = S3DIS_LABELS.get(class_name, S3DIS_LABELS["clutter"])
            points = np.loadtxt(annotation_file, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            if points.shape[1] < 6:
                continue

            coordinates = points[:, :3]
            rgb = np.clip(points[:, 3:6], 0, 255) / 255.0
            labels = np.ones(coordinates.shape[0], dtype=np.int64) * obj_id

            coordinates_list.append(coordinates)
            rgb_list.append(rgb)
            labels_list.append(labels)
            obj2label_map[str(obj_id)] = semantic_id
            click_idx[str(obj_id)] = []
            obj_id += 1

        if not coordinates_list:
            raise RuntimeError(f"No valid S3DIS annotations found for {room['area']}/{room['room']}")

        coordinates = np.vstack(coordinates_list).astype(np.float32)
        rgb = np.vstack(rgb_list).astype(np.float32)
        labels = np.hstack(labels_list).astype(np.int64)

        if self.max_points_per_room is not None and coordinates.shape[0] > self.max_points_per_room:
            selected = np.random.choice(coordinates.shape[0], self.max_points_per_room, replace=False)
            coordinates = coordinates[selected]
            rgb = rgb[selected]
            labels = labels[selected]

        if self.center_coordinates:
            coordinates -= coordinates.mean(0)

        # Reuse the existing 2-channel sweep=1 backbone input path:
        # the collate function drops the time column and keeps intensity + distance.
        intensity = rgb.mean(axis=1, keepdims=True)
        time_array = np.zeros((coordinates.shape[0], 1), dtype=np.float32)
        features = np.hstack((time_array, intensity))

        center_coordinate = coordinates.mean(0)
        distance = np.linalg.norm(coordinates - center_coordinate, axis=1)[:, np.newaxis]
        features = np.hstack((features, distance))

        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        features = np.hstack((coordinates, features))
        shuffle_order = np.arange(labels.shape[0])
        np.random.shuffle(shuffle_order)

        coordinates = coordinates[shuffle_order]
        features = features[shuffle_order]
        labels = labels[shuffle_order]

        return {
            "sequence": [os.path.join(room["area"], room["room"])],
            "num_points": [len(labels)],
            "num_obj": [len(obj2label_map)],
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "click_idx": click_idx,
            "obj2label": [obj2label_map],
        }
