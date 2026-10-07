"""Dataset loading utilities.

Default behavior reproduces the manuscript-era image-level 70/15/15 split.
For publication-grade re-runs, prefer a patient/group-level split whenever a
patient identifier is available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def split_pairs(images: Sequence[str], masks: Sequence[str], split=0.15, seed=42):
    """Reproduce the original image-wise split, but split paired records together."""
    if len(images) != len(masks):
        raise ValueError(f"images ({len(images)}) and masks ({len(masks)}) differ")
    pairs = list(zip(images, masks))
    valid_size = int(split * len(pairs))
    test_size = int(split * len(pairs))
    train_pairs, valid_pairs = train_test_split(
        pairs, test_size=valid_size, random_state=seed
    )
    train_pairs, test_pairs = train_test_split(
        train_pairs, test_size=test_size, random_state=seed
    )
    unzip = lambda xs: tuple(map(list, zip(*xs))) if xs else ([], [])
    return unzip(train_pairs), unzip(valid_pairs), unzip(test_pairs)


def _read_image(path, image_size=256, channels=3):
    path = path.decode() if isinstance(path, (bytes, bytearray)) else str(path)
    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    x = cv2.imread(path, flag)
    if x is None:
        raise FileNotFoundError(path)
    x = cv2.resize(x, (image_size, image_size)).astype(np.float32) / 255.0
    if channels == 1:
        x = x[..., None]
    return x


def _read_mask(path, image_size=256):
    path = path.decode() if isinstance(path, (bytes, bytearray)) else str(path)
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if y is None:
        raise FileNotFoundError(path)
    y = cv2.resize(y, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    y = (y.astype(np.float32) / 255.0)[..., None]
    return y


def make_dataset(images, masks, batch_size=8, image_size=256, channels=3,
                 shuffle=False, repeat=False, seed=42):
    def parse(x, y):
        xi, yi = tf.numpy_function(
            lambda a, b: (_read_image(a, image_size, channels), _read_mask(b, image_size)),
            [x, y], [tf.float32, tf.float32]
        )
        xi.set_shape([image_size, image_size, channels])
        yi.set_shape([image_size, image_size, 1])
        return xi, yi

    ds = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)
