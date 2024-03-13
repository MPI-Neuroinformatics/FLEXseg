#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module aims to apply FLEXseg.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# change cwd to directory of this file
os.chdir(Path(__file__).parent.resolve())  # nopep8

# custom packages
from image_operations.transform_image import (
    transform_image,
)
from use_gpu import set_up_cuda
from image_operations.correction import (
    clean_up_image,
)
from image_operations.itk_transform import (
    affine_transform_image,
)
from model_predictions import (
    initialize_cropped_generator,
    make_model_prediction,
)
from model.resnet import ResMirror

# -----------------------------------------------------------------------------
# Paths and Definitions
# -----------------------------------------------------------------------------
IMAGE_RFP = "image.nii"
IMAGE_0P6MM_RFP = "image_0p6mm.nii.gz"
SEGMENTATION_0P6MM_RFP = "segmentation_0p6mm.nii.gz"
SEGMENTATION_RFP = "segmentation.nii.gz"

TARGET_AFFINE = .6 * np.eye(3)

MODEL_WEIGHTS_AFP = (
    Path.cwd().parent
    / "model_weights/f8472e3be0f14e6d8302b6acfdc6c0bb_segmentation.pt"
)
NUM_TYPES = 5
# Types = {
#     0: "background",
#     1: "cortical_gm_left",
#     2: "cortical_gm_right",
#     3: "cerebral_wm_left",
#     4: "cerebral_wm_right",
# }
MODEL_INPUT_SHAPE = [128, 128, 128]
STANDARDIZATION_SHAPE = [384, 384, 384]


# Hardware
DEVICE = set_up_cuda()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Build model
generator = ResMirror(
    input_channels=1,
    output_channels=NUM_TYPES,
    num_planes=48,
    name_block="bottleneck",
    num_blocks=[3, 4, 23, 3],
    activation="relu",
    variant="original",
)
generator.to(DEVICE)
generator.load_state_dict(torch.load(MODEL_WEIGHTS_AFP))

final_activation_fn = nn.LogSoftmax(dim=1)

model = nn.Sequential(generator, final_activation_fn)
model.eval()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    subject = Path(args.subject)

    # Check input path
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        raise ValueError(f"Input does not exist: {input_path}")

    # Check output destination
    if args.output_path == "":
        working_dir_adp = input_path.parent / subject
    elif output_path.is_dir():
        working_dir_adp = output_path / subject
    elif not output_path.is_absolute():
        working_dir_adp = Path.cwd() / output_path / subject
    else:
        raise ValueError(
            f"Ther is no valid output path given: {args.output_path}")
    working_dir_adp.mkdir(parents=False, exist_ok=True)

    # Define constants
    image_afp = working_dir_adp / IMAGE_RFP
    image_world_space_afp = working_dir_adp / IMAGE_0P6MM_RFP
    segmentation_world_space_afp = working_dir_adp / SEGMENTATION_0P6MM_RFP
    segmentation_afp = working_dir_adp / SEGMENTATION_RFP

    # -------------------------------------------------------------------------
    print("Preparing image...")
    # -------------------------------------------------------------------------

    # Check for strange paths
    if not image_afp.is_file():
        if input_path.suffix == ".nii":
            image_afp = image_afp.with_suffix("")
        elif input_path.suffix == ".gz":
            image_afp = image_afp.with_suffix(".nii.gz")
        elif input_path.suffix == ".mgz":
            image_afp = image_afp.with_suffix("").with_suffix(".mgz")
        else:
            raise ValueError(
                f"Extension not supported: {input_path.suffixes}")
        if not image_afp.is_file():
            try:
                image_afp.symlink_to(input_path)
            except OSError:
                image_afp = input_path

    image_nii = nib.load(image_afp)
    image_raw = image_nii.dataobj[:]
    affine_raw = image_nii.affine

    # -------------------------------------------------------------------------
    print("Preprocessing...")
    # -------------------------------------------------------------------------
    if not image_world_space_afp.is_file() or args.overwrite:
        # clean up
        image_cleaned = clean_up_image(image_raw)

        # transform
        if not np.allclose(affine_raw[:3, :3], TARGET_AFFINE, atol=1e-3):
            do_transform = True
            image_world_space, world_space_affine = affine_transform_image(
                image=image_cleaned.astype(np.float32),
                affine_matrix=affine_raw,
                target_affine_matrix=TARGET_AFFINE,
                interpolation='linear',
            )
        else:
            do_transform = False
            image_world_space = image_cleaned.copy().astype(np.float32)
            world_space_affine = affine_raw.copy()

        # standardize
        img_numel = np.prod(image_world_space.shape)
        norm_numel = np.prod(STANDARDIZATION_SHAPE)

        mean = np.mean(np.append(
            image_world_space.flatten(),
            np.zeros(norm_numel - img_numel),
        ))
        std = np.std(np.append(
            image_world_space.flatten(),
            np.zeros(norm_numel - img_numel),
        ))
        image_world_space = np.divide(
            np.subtract(image_world_space, mean, out=image_world_space),
            std,
            out=image_world_space
        ).astype(np.float32)

        image_world_space_nii = nib.Nifti1Image(
            image_world_space,
            world_space_affine,
            dtype=np.float32,
        )
        nib.save(image_world_space_nii, image_world_space_afp)

    else:
        image_world_space_nii = nib.load(image_world_space_afp)
        image_world_space = image_world_space_nii.dataobj[:]
        world_space_affine = image_world_space_nii.affine
        do_transform = True

    image_world_space = image_world_space.astype(np.float32, copy=False)

    # -------------------------------------------------------------------------
    print("Segmenting...")
    # -------------------------------------------------------------------------
    data_generator = initialize_cropped_generator(
        image=torch.from_numpy(image_world_space),
        batch_size=1,
        num_cpu_workers=0,
        num_crops_per_direction=9,
        crop_size=tuple(MODEL_INPUT_SHAPE),
    )

    # Prediction
    with torch.no_grad():
        y_pred_mask = torch.zeros(
            (1, NUM_TYPES) + image_world_space.shape,
            dtype=torch.float,
            requires_grad=False,
            device=DEVICE,
        )
        y_pred_mask = make_model_prediction(
            data_generator=data_generator,
            model=model,
            prediction=y_pred_mask,
        )
        y_pred_mask = torch.argmax(y_pred_mask, dim=1)
        y_pred_mask = y_pred_mask.detach().cpu().numpy().astype(np.int16)

    y_pred_mask = np.squeeze(y_pred_mask).astype(np.int16)

    # Saving
    segmentation_world_space_nii = nib.Nifti1Image(
        y_pred_mask,
        world_space_affine,
        dtype=np.int16,
    )
    nib.save(segmentation_world_space_nii, segmentation_world_space_afp)

    # -------------------------------------------------------------------------
    print("Postprocessing...")
    # -------------------------------------------------------------------------
    if do_transform:
        seg_transformed, _ = transform_image(
            image=y_pred_mask.astype(np.uint8),
            affine_matrix=world_space_affine,
            target_affine_matrix=affine_raw,
            target_shape=image_raw.shape,
            interpolation="majority_voted",
        )
        segmentation_nii = nib.Nifti1Image(
            seg_transformed.astype(np.int16),
            affine_raw,
            dtype=np.int16,
        )
        nib.save(segmentation_nii, segmentation_afp)

    torch.cuda.empty_cache()


# %% Execute
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FLEXseg Tissue Segmentation Script')
    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Enter the subject identification string for output.',
    )
    parser.add_argument(
        '--output_path',
        default='',
        type=str,
        help='Enter output path where to save results.',
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Specify input with path to a NIfTI file or DICOM folder.',
    )
    parser.add_argument(
        '--overwrite',
        action="store_true",
        help="Set flag to overwrite possibly existing files.",
    )

    arguments = parser.parse_args()

    main(arguments)
