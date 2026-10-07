#!/usr/bin/env python3
"""Minimal reproducible trainer for baseline or Portable-Bridge U-Net."""
import argparse
from glob import glob
from pathlib import Path
import tensorflow as tf
from portable_bridge_unet.data import split_pairs, make_dataset
from portable_bridge_unet.metrics import dice_coef, jaccard_coef
from portable_bridge_unet.models import build_unet, build_portable_bridge_unet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True, help='Glob for input images')
    p.add_argument('--masks', required=True, help='Glob for masks')
    p.add_argument('--model', choices=['unet','portable_bridge'], default='portable_bridge')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--channels', type=int, choices=[1,3], default=3)
    p.add_argument('--out', default='runs/model.keras')
    args = p.parse_args()

    images, masks = sorted(glob(args.images)), sorted(glob(args.masks))
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_pairs(images, masks)
    print(f'train={len(train_x)} val={len(val_x)} test={len(test_x)}')

    train_ds = make_dataset(train_x, train_y, args.batch_size, channels=args.channels, shuffle=True)
    val_ds = make_dataset(val_x, val_y, args.batch_size, channels=args.channels)

    builder = build_unet if args.model == 'unet' else build_portable_bridge_unet
    model = builder(channels=args.channels)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='binary_crossentropy',
        metrics=[dice_coef, jaccard_coef, tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 'accuracy'],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=30),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)


if __name__ == '__main__':
    main()
