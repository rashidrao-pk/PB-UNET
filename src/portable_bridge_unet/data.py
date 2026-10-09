from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import csv
import random

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Sample:
    image: str
    mask: str
    group: str | None = None


def load_samples_csv(path: str | Path) -> list[Sample]:
    """
    Load segmentation samples from a CSV manifest.

    Required columns:
        image
        mask

    Optional column:
        group

    Extra columns such as class, image_id, etc. are ignored.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Split CSV not found: {path}")

    samples: list[Sample] = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        required = {"image", "mask"}
        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"{path} missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            mask = (row.get("mask") or "").strip()
            group = (row.get("group") or "").strip() or None

            if not image or not mask:
                raise ValueError(
                    f"Missing image/mask path in {path} at row {row_number}"
                )

            if not Path(image).exists():
                raise FileNotFoundError(
                    f"Image from split CSV does not exist: {image}"
                )

            if not Path(mask).exists():
                raise FileNotFoundError(
                    f"Mask from split CSV does not exist: {mask}"
                )

            samples.append(
                Sample(
                    image=image,
                    mask=mask,
                    group=group,
                )
            )

    if not samples:
        raise ValueError(f"No samples found in split CSV: {path}")

    return samples


def pair_by_stem(
    images: Sequence[str],
    masks: Sequence[str],
) -> list[Sample]:
    """
    Pair images and masks by filename stem rather than sorted index.
    """
    image_map = {Path(p).stem: str(p) for p in images}
    mask_map = {Path(p).stem: str(p) for p in masks}

    if len(image_map) != len(images) or len(mask_map) != len(masks):
        raise ValueError(
            "duplicate filename stems; use unique patient-prefixed filenames"
        )

    common = sorted(image_map.keys() & mask_map.keys())

    if not common:
        raise ValueError(
            "no matching image/mask filename stems found"
        )

    missing_i = sorted(mask_map.keys() - image_map.keys())
    missing_m = sorted(image_map.keys() - mask_map.keys())

    if missing_i or missing_m:
        raise ValueError(
            f"unpaired files: masks_without_images={len(missing_i)}, "
            f"images_without_masks={len(missing_m)}"
        )

    return [
        Sample(image_map[k], mask_map[k])
        for k in common
    ]


def split_samples_legacy(
    samples: Sequence[Sample],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """
    Reproduce the original image-level split semantics.

    This remains available for backward compatibility with Sunnybrook.
    """
    samples = list(samples)

    n_val = int(val_fraction * len(samples))
    n_test = int(test_fraction * len(samples))

    train, val = train_test_split(
        samples,
        test_size=n_val,
        random_state=seed,
    )

    train, test = train_test_split(
        train,
        test_size=n_test,
        random_state=seed,
    )

    return list(train), list(val), list(test)


def split_samples_grouped(
    samples: Sequence[Sample],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """
    Patient/group-level split.

    Recommended when reliable patient/group identifiers are available.
    """
    samples = list(samples)

    if any(s.group is None for s in samples):
        raise ValueError(
            "all samples need a non-null group for grouped splitting"
        )

    groups = np.asarray([s.group for s in samples])
    idx = np.arange(len(samples))

    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        gss_test.split(idx, groups=groups)
    )

    remaining_val_fraction = (
        val_fraction / (1.0 - test_fraction)
    )

    remaining_groups = groups[train_val_idx]

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=remaining_val_fraction,
        random_state=seed,
    )

    train_rel, val_rel = next(
        gss_val.split(
            train_val_idx,
            groups=remaining_groups,
        )
    )

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    def pick(ids):
        return [samples[int(i)] for i in ids]

    return (
        pick(train_idx),
        pick(val_idx),
        pick(test_idx),
    )


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 256,
        channels: int = 3,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.augment = bool(augment)

        if self.channels not in (1, 3):
            raise ValueError(
                "channels must be 1 or 3"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _read(
        self,
        sample: Sample,
    ) -> tuple[np.ndarray, np.ndarray]:

        flag = (
            cv2.IMREAD_GRAYSCALE
            if self.channels == 1
            else cv2.IMREAD_COLOR
        )

        image = cv2.imread(sample.image, flag)
        mask = cv2.imread(
            sample.mask,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(sample.image)

        if mask is None:
            raise FileNotFoundError(sample.mask)

        if self.channels == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(
                    image,
                    axis=1,
                ).copy()

                mask = np.flip(
                    mask,
                    axis=1,
                ).copy()

            if random.random() < 0.5:
                image = np.flip(
                    image,
                    axis=0,
                ).copy()

                mask = np.flip(
                    mask,
                    axis=0,
                ).copy()

        image = image.astype(np.float32) / 255.0

        mask = (
            mask.astype(np.float32) / 255.0 > 0.5
        ).astype(np.float32)

        if self.channels == 1:
            image = image[None, ...]
        else:
            image = np.transpose(
                image,
                (2, 0, 1),
            )

        mask = mask[None, ...]

        return image, mask

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:

        sample = self.samples[index]

        image, mask = self._read(sample)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "image_path": sample.image,
            "mask_path": sample.mask,
            "group": sample.group or "",
        }


def make_loader(
    samples: Sequence[Sample],
    batch_size: int,
    image_size: int = 256,
    channels: int = 3,
    shuffle: bool = False,
    augment: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:

    ds = SegmentationDataset(
        samples,
        image_size=image_size,
        channels=channels,
        augment=augment,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )