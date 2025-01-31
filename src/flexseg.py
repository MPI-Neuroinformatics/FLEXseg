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
import time
from typing import Callable, Tuple
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# change cwd to directory of this file include submodules
os.chdir(Path(__file__).parent.resolve())  # nopep8
sys.path.append(str(Path.cwd().parent / "src"))  # nopep8
sys.path.append(str(Path.cwd().parent / "mrimage_processing"))  # nopep8

# custom packages
from mrimage_processing.data_flow.nifti_operations import read_nii, save_nii
from mrimage_processing.intensity_modification.correction import (
    clean_up_image,
)
from mrimage_processing.spatial_transformation.itk_transform import (
    affine_transform_image
)
from mrimage_processing.spatial_transformation.np_transform import (
    affine_transform_image as custom_transform_image
)

from use_gpu import set_up_cuda

from model_predictions import (
    initialize_cropped_generator,
    make_model_prediction,
)

# experiment package
# initial FLEXseg
from exp_runs import exp_f8472e3be0f14e6d8302b6acfdc6c0bb_params
# with CSF, weighted classes
from exp_runs import exp_f210acd3840c41f79631547e41ecbac5_params
# with CSF, weighted classes and warm up discriminator
from exp_runs import exp_fdc29906ff3344bb914b7575c9dd1f91_params
# unet
from exp_runs import exp_6be8693956564eea9b6a14dc0714012a_params
# convnext
from exp_runs import exp_62120809342741c8b32e0b04a3ec7296_params


# -----------------------------------------------------------------------------
# Paths and Definitions
# -----------------------------------------------------------------------------
IMAGE_RFP = "image.nii"
IMAGE_0P6MM_RFP = "image_0p6mm.nii.gz"
SEGMENTATION_0P6MM_RFP = "segmentation_0p6mm.nii.gz"
SEGMENTATION_RFP = "segmentation.nii.gz"

MODEL_INPUT_SHAPE = [128, 128, 128]

# Hardware
if os.environ['CONDA_DEFAULT_ENV'] == "FLEXseg_cpu":
    DEVICE = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    DEVICE = set_up_cuda()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Build model


def pick_model(
        run_id: str,
        probabilities: bool = False,
) -> Tuple[nn.Module, int, Callable]:
    """
    Pick model from run id.

    Parameters
    ----------
    run_id : str
        Experiment run ID.
    probabilities : bool, optional
        If the model should give probabilities per type per voxel. The default
        is False.

    Raises
    ------
    ValueError
        If no valid model selected.

    Returns
    -------
    model : nn.Module
        Model to predict segmentation.
    num_types : int
        Number of types that should be predicted.
    preprocessing_fn : Callable
        Function to preprocess MR image according to experimental setup.

    """
    if run_id == "f8472e3be0f14e6d8302b6acfdc6c0bb":
        print("Original FLEXseg")
        exp_params = exp_f8472e3be0f14e6d8302b6acfdc6c0bb_params
    elif run_id == "fdc29906ff3344bb914b7575c9dd1f91":
        print("FLEXseg with pretraining")
        exp_params = exp_f210acd3840c41f79631547e41ecbac5_params
    elif run_id == "fdc29906ff3344bb914b7575c9dd1f91":
        print("FLEXseg with pretraining and warm up")
        exp_params = exp_fdc29906ff3344bb914b7575c9dd1f91_params
    elif run_id == "fdc29906ff3344bb914b7575c9dd1f91":
        print("FLEXseg with convnext")
        exp_params = exp_62120809342741c8b32e0b04a3ec7296_params
    elif run_id == "fdc29906ff3344bb914b7575c9dd1f91":
        print("FLEXseg with unet")
        exp_params = exp_6be8693956564eea9b6a14dc0714012a_params
    else:
        raise ValueError("No valid model choice.")

    generator = exp_params.generator
    generator.to(DEVICE)
    generator.load_state_dict(torch.load(
        Path.cwd().parent / exp_params.MODEL_WEIGHTS_RFP,
        map_location=DEVICE),
    )

    if probabilities:
        final_activation_fn = nn.Softmax(dim=1)
    else:
        final_activation_fn = nn.LogSoftmax(dim=1)
    model = nn.Sequential(generator, final_activation_fn)
    model.eval()

    return model, exp_params.NUM_TYPES, exp_params.preprocessing


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(args):

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    subject = Path(args.subject)

    model, num_types, preprocessing = pick_model(
        args.model_run_id, args.store_probabilities)

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

    if args.ultra_high:
        TARGET_AFFINE = .3 * np.eye(3)
        NUM_CROPS_PER_DIRECTION = 18
    else:
        TARGET_AFFINE = .6 * np.eye(3)
        NUM_CROPS_PER_DIRECTION = 9

    # -------------------------------------------------------------------------
    print("Preparing image...")
    # -------------------------------------------------------------------------
    t_0 = time.time()
    # Check for strange paths
    if not image_afp.is_file():
        if input_path.suffix == ".nii":
            image_afp = image_afp.with_suffix(".nii")
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

    image_raw, affine_raw = read_nii(image_afp)

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

        # experiment specific preprocessing
        image_world_space, world_space_affine = preprocessing(
            image_world_space, world_space_affine
        )
        save_nii(
            image=image_world_space,
            affine_matrix=world_space_affine,
            filepath=image_world_space_afp,
            dtype=np.float32,
        )

    else:
        image_world_space, world_space_affine = read_nii(image_world_space_afp)
        do_transform = True

    image_world_space = image_world_space.astype(np.float32, copy=False)

    print(f"Preprocessing done in {time.time() - t_0 :.1f} s.")
    # -------------------------------------------------------------------------
    print("Segmenting...")
    # -------------------------------------------------------------------------
    t_1 = time.time()
    if (not segmentation_world_space_afp.is_file() or args.overwrite):
        data_generator = initialize_cropped_generator(
            image=torch.from_numpy(image_world_space),
            batch_size=1,
            num_cpu_workers=0,
            num_crops_per_direction=NUM_CROPS_PER_DIRECTION,
            crop_size=tuple(MODEL_INPUT_SHAPE),
        )

        # Prediction
        with torch.no_grad():
            y_pred_mask = torch.zeros(
                (1, num_types) + image_world_space.shape,
                dtype=torch.float,
                requires_grad=False,
                device=DEVICE,
            )
            y_pred_mask = make_model_prediction(
                data_generator=data_generator,
                model=model,
                prediction=y_pred_mask,
                normalize=False,
            )

            # Store probabilities
            if args.store_probabilities:
                y_probabilities = y_pred_mask[0].detach(
                ).cpu().numpy().astype(float)
                for label in range(num_types):
                    y_probability = y_probabilities[label]
                    save_nii(
                        image=y_probability,
                        affine_matrix=world_space_affine,
                        filepath=segmentation_world_space_afp.with_stem(
                            f"segmentation_0p6mm_{label}.nii"
                        ),
                        dtype=np.float32,
                    )

            y_pred_mask = torch.argmax(y_pred_mask, dim=1)
            y_pred_mask = y_pred_mask.detach().cpu().numpy().astype(np.int16)

        y_pred_mask = np.squeeze(y_pred_mask).astype(np.int16)

        # Saving
        save_nii(
            image=y_pred_mask,
            affine_matrix=world_space_affine,
            filepath=segmentation_world_space_afp,
            dtype=np.int16,
        )
    else:
        y_pred_mask, _ = read_nii(segmentation_world_space_afp)

    print(f"Segmentation done in {time.time() - t_1 :.1f} s.")
    # -------------------------------------------------------------------------
    print("Postprocessing...")
    # -------------------------------------------------------------------------
    if (not segmentation_afp.is_file() or args.overwrite) and do_transform:
        if args.use_attentive_postprocessing:
            seg_transformed, _ = custom_transform_image(
                image=y_pred_mask.astype(np.uint8),
                affine_matrix=world_space_affine,
                target_affine_matrix=affine_raw,
                target_shape=image_raw.shape,
                interpolation="majority_voted",
            )
        else:
            seg_transformed, _ = affine_transform_image(
                image=y_pred_mask.astype(float),
                affine_matrix=world_space_affine,
                target_affine_matrix=affine_raw,
                target_shape=image_raw.shape,
                interpolation='nearest',
            )
            seg_transformed = nib.casting.float_to_int(
                seg_transformed, "int16")

        save_nii(
            image=seg_transformed.astype(np.int16),
            affine_matrix=affine_raw,
            filepath=segmentation_afp,
            dtype=np.int16,
        )

    print(f"FLEXseg done in {time.time() - t_0 :.1f} s.")

    # disable emptying cache for multiple images to save ~45s.
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
    parser.add_argument(
        '--store_probabilities',
        action="store_true",
        help="Set flag to save probability masks.",
    )
    parser.add_argument(
        '--ultra_high',
        action='store_true',
        help="Set flag to segment on doubled resolution.",
    )
    parser.add_argument(
        '--use_attentive_postprocessing',
        action="store_true",
        help="Set flag to use attentive but time consuming transformation.",
    )
    parser.add_argument(
        '--model_run_id',
        default='fdc29906ff3344bb914b7575c9dd1f91',
        type=str,
        help='Specify which model and parameter set to take.',
    )

    arguments = parser.parse_args()

    main(arguments)
