#!/usr/bin/env python
#
# Estimate a betaseries with each object in each run for an ROI.

import os
import argparse
import numpy as np
import nibabel as nib
from tesser import rsa


def main(
    raw_dir,
    post_dir,
    mask,
    bold,
    subject,
    space='T1w',
    save_format='matrix',
    mask_dir='func',
    mask_thresh=0.001,
):
    # run betaseries estimation for each run
    high_pass = 1 / 128
    beta = np.vstack(
        [
            rsa.run_betaseries(
                raw_dir,
                post_dir,
                mask,
                bold,
                subject,
                run,
                high_pass,
                space,
                mask_dir,
                mask_thresh,
            ) for run in range(1, 7)
        ]
    )

    # save a numpy array with the results
    out_dir = os.path.join(post_dir, 'results', 'beta', bold, mask)
    os.makedirs(out_dir, exist_ok=True)
    if save_format == 'matrix':
        out_file = os.path.join(out_dir, f'beta_{subject}.npy')
        np.save(out_file, beta)
    elif save_format == 'image':
        run = 1
        if mask_dir == 'func':
            mask_file = os.path.join(
                post_dir,
                f'sub-{subject}',
                'func',
                f'sub-{subject}_task-struct_run-{run}_space-{space}_desc-{mask}_mask.nii.gz',
            )
        else:
            mask_file = os.path.join(
                post_dir,
                f'sub-{subject}',
                mask_dir,
                f'sub-{subject}_space-{space}_desc-{mask}_mask.nii.gz',
            )
        mask_vol = nib.load(mask_file)
        if mask_thresh is None:
            mask_img = mask_vol.get_fdata().astype(bool)
        else:
            mask_img = mask_vol.get_fdata() > mask_thresh
        out_data = np.zeros([*mask_img.shape, beta.shape[0]])
        out_data[mask_img, :] = beta.T
        new_img = nib.Nifti1Image(out_data, mask_vol.affine, mask_vol.header)
        out_file = os.path.join(out_dir, f'beta_{subject}.nii.gz')
        nib.save(new_img, out_file)
    else:
        raise ValueError(f'Invalid save format: {save_format}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw data BIDS directory')
    parser.add_argument('post_dir', help='path to directory with model results')
    parser.add_argument('mask', help='desc of mask defining the ROI')
    parser.add_argument('bold', help='desc of preprocessed bold images')
    parser.add_argument('subject', help='subject identifier (e.g., 100)')
    parser.add_argument('--space', '-s', default='T1w', help='image space to use')
    parser.add_argument(
        '--format', '-f', default='matrix', help='output format [("matrix"), "image"]'
    )
    parser.add_argument(
        '--mask-dir', '-m', help='directory with the mask [("func"), "anat"]'
    )
    parser.add_argument(
        '--mask-thresh', '-t', type=float, help='threshold to apply to the mask'
    )
    args = parser.parse_args()
    main(
        args.raw_dir,
        args.post_dir,
        args.mask,
        args.bold,
        args.subject,
        args.space,
        args.format,
        args.mask_dir,
        args.mask_thresh,
    )
