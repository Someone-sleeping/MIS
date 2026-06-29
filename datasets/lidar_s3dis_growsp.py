import os
from pathlib import Path
from random import random
from typing import Optional

import MinkowskiEngine as ME
import numpy as np
import volumentations as V
from torch.utils.data import Dataset

from interactive_tool.utils import read_ply


class GrowSPPseudoS3DISDataset(Dataset):
    """S3DIS training data supervised only by frozen GrowSP pseudo labels.

    GrowSP pseudo labels are stored after GrowSP's own 0.05m sparse quantization,
    so this dataset reproduces that quantization before exposing samples to the
    Interactive4D collate function.
    """

    def __init__(
        self,
        growsp_data_dir: str,
        pseudo_label_dir: str,
        mode: Optional[str] = "train",
        volume_augmentations_path: Optional[str] = None,
        sweep: Optional[int] = 1,
        center_coordinates=True,
        validation_area: str = "Area_5",
        max_points_per_room: Optional[int] = 9000,
        growsp_voxel_size: float = 0.05,
        clip_bound: Optional[float] = 4.0,
        pseudo_semantic_label: int = 13,
        min_points_per_object: int = 5,
    ):
        super().__init__()
        if sweep != 1:
            raise ValueError("GrowSP pseudo S3DIS currently supports only sweep=1.")

        self.mode = mode
        self.growsp_data_dir = Path(growsp_data_dir)
        self.pseudo_label_dir = Path(pseudo_label_dir)
        self.center_coordinates = center_coordinates
        self.validation_area = validation_area
        self.max_points_per_room = max_points_per_room
        self.growsp_voxel_size = growsp_voxel_size
        self.clip_bound = clip_bound
        self.pseudo_semantic_label = pseudo_semantic_label
        self.min_points_per_object = min_points_per_object

        if not self.growsp_data_dir.exists():
            raise FileNotFoundError(f"GrowSP S3DIS input directory not found: {self.growsp_data_dir}")
        if not self.pseudo_label_dir.exists():
            raise FileNotFoundError(f"GrowSP pseudo label directory not found: {self.pseudo_label_dir}")

        self.data = self._discover_rooms()
        if len(self.data) == 0:
            raise FileNotFoundError(
                f"No matched GrowSP S3DIS PLY/pseudo pairs found under {self.growsp_data_dir} and {self.pseudo_label_dir}"
            )

        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(volume_augmentations_path, data_format="yaml")

    def _discover_rooms(self):
        rooms = []
        for ply_path in sorted(self.growsp_data_dir.glob("*.ply")):
            scene_name = ply_path.stem
            area = scene_name.split("_", 2)[0] + "_" + scene_name.split("_", 2)[1]
            is_validation = area == self.validation_area
            if ("train" in self.mode and is_validation) or ("validation" in self.mode and not is_validation):
                continue
            pseudo_path = self.pseudo_label_dir / f"{scene_name}.npy"
            if pseudo_path.exists():
                rooms.append({"scene_name": scene_name, "area": area, "ply_path": ply_path, "pseudo_path": pseudo_path})
        return rooms

    def __len__(self):
        return len(self.data)

    def _clip(self, coords):
        if self.clip_bound is None:
            return None
        bound_min = np.min(coords, axis=0).astype(float)
        bound_max = np.max(coords, axis=0).astype(float)
        bound_size = bound_max - bound_min
        if bound_size.max() < self.clip_bound:
            return None
        center = bound_min + bound_size * 0.5
        lim = float(self.clip_bound)
        return (
            (coords[:, 0] >= (-lim + center[0]))
            & (coords[:, 0] < (lim + center[0]))
            & (coords[:, 1] >= (-lim + center[1]))
            & (coords[:, 1] < (lim + center[1]))
            & (coords[:, 2] >= (-lim + center[2]))
            & (coords[:, 2] < (lim + center[2]))
        )

    def _growsp_quantize(self, coords, colors):
        clip_inds = self._clip(coords)
        if clip_inds is not None:
            coords = coords[clip_inds]
            colors = colors[clip_inds]

        quantized = np.floor(coords / self.growsp_voxel_size)
        _, _, unique_map, _ = ME.utils.sparse_quantize(
            coordinates=np.ascontiguousarray(quantized),
            features=colors,
            return_index=True,
            return_inverse=True,
        )
        return coords[unique_map], colors[unique_map]

    def _relabel_pseudo_objects(self, pseudo):
        labels = np.zeros(pseudo.shape[0], dtype=np.int64)
        for pseudo_id, obj_id in self._valid_object_ids(pseudo):
            labels[pseudo == pseudo_id] = obj_id
        return self._compact_object_ids(labels)

    def _valid_object_ids(self, labels):
        obj_id = 1
        for source_id in sorted(int(v) for v in np.unique(labels) if int(v) >= 0):
            mask = labels == source_id
            if int(mask.sum()) < self.min_points_per_object:
                continue
            yield source_id, obj_id
            obj_id += 1

    def _compact_object_ids(self, labels):
        compact = np.zeros(labels.shape[0], dtype=np.int64)
        obj2label_map = {}
        click_idx = {"0": []}
        obj_id = 1
        for source_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            mask = labels == source_id
            if int(mask.sum()) < self.min_points_per_object:
                continue
            compact[mask] = obj_id
            obj2label_map[str(obj_id)] = self.pseudo_semantic_label
            click_idx[str(obj_id)] = []
            obj_id += 1
        return compact, obj2label_map, click_idx

    def __getitem__(self, idx):
        room = self.data[idx]
        data = read_ply(str(room["ply_path"]))
        coords = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
        colors = np.vstack((data["red"], data["green"], data["blue"])).T.astype(np.float32)
        coords -= coords.mean(0)

        coords, colors = self._growsp_quantize(coords, colors)
        pseudo = np.asarray(np.load(room["pseudo_path"]), dtype=np.int64)
        if pseudo.shape[0] != coords.shape[0]:
            raise ValueError(
                f"GrowSP pseudo length mismatch for {room['scene_name']}: "
                f"pseudo={pseudo.shape[0]} quantized_points={coords.shape[0]}"
            )

        labels, obj2label_map, click_idx = self._relabel_pseudo_objects(pseudo)
        valid = labels > 0
        if valid.sum() == 0:
            raise RuntimeError(f"No valid GrowSP pseudo objects found for {room['scene_name']}")

        if self.max_points_per_room is not None and coords.shape[0] > self.max_points_per_room:
            selected = np.random.choice(coords.shape[0], self.max_points_per_room, replace=False)
            coords = coords[selected]
            colors = colors[selected]
            labels = labels[selected]
            labels, obj2label_map, click_idx = self._compact_object_ids(labels)

        if self.center_coordinates:
            coords -= coords.mean(0)

        rgb = np.clip(colors, 0, 255) / 255.0
        intensity = rgb.mean(axis=1, keepdims=True)
        time_array = np.zeros((coords.shape[0], 1), dtype=np.float32)
        center_coordinate = coords.mean(0)
        distance = np.linalg.norm(coords - center_coordinate, axis=1)[:, np.newaxis]
        features = np.hstack((time_array, intensity, distance))

        if "train" in self.mode:
            coords -= coords.mean(0)
            if 0.5 > random():
                coords += np.random.uniform(coords.min(0), coords.max(0)) / 2
            aug = self.volume_augmentations(points=coords)
            coords = aug["points"]

        features = np.hstack((coords, features))
        shuffle_order = np.arange(labels.shape[0])
        np.random.shuffle(shuffle_order)
        coords = coords[shuffle_order]
        features = features[shuffle_order]
        labels = labels[shuffle_order]

        return {
            "sequence": [room["scene_name"]],
            "num_points": [len(labels)],
            "num_obj": [len(obj2label_map)],
            "coordinates": coords.astype(np.float32),
            "features": features.astype(np.float32),
            "labels": labels.astype(np.int64),
            "click_idx": click_idx,
            "obj2label": [obj2label_map],
        }
