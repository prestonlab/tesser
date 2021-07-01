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
    exclude_motion=False,
):
    # run betaseries estimation for each run
    high_pass = 1 / 128
    beta = None
    resid = {}
    for run in range(1, 7):
        run_beta, run_resid = rsa.run_betaseries(
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
            exclude_motion,
        )
        if beta is None:
            beta = run_beta
        else:
            beta = np.vstack((beta, run_beta))
        resid[run] = run_resid

    # save a numpy array with the results
    beta = 'beta'
    if exclude_motion:
        beta += 'MO'
    out_dir = os.path.join(post_dir, 'results', beta, bold, mask, f'sub-{subject}')
    os.makedirs(out_dir, exist_ok=True)
    if save_format == 'matrix':
        np.save(os.path.join(out_dir, f'sub-{subject}_beta.npy'), beta)
    elif save_format == 'image':
        run = 1
        if mask_dir == 'func':
            mask_file = rsa.get_func_mask(
                post_dir, subject, 'struct', run, space, desc=mask
            )
        else:
            mask_file = rsa.get_anat_mask(post_dir, subject, space, label=mask)
        mask_vol = nib.load(mask_file)
        if mask_thresh is None:
            mask_img = mask_vol.get_fdata().astype(bool)
        else:
            mask_img = mask_vol.get_fdata() > mask_thresh

        # save events
        vols = rsa.load_struct_vols(raw_dir, subject)
        events = vols.copy()
        events['onset'] = np.arange(len(events))
        events['duration'] = 1.0
        events.to_csv(
            os.path.join(out_dir, f'sub-{subject}_events.tsv'), sep='\t', index=False
        )

        # save the mask
        new_img = nib.Nifti1Image(mask_img, mask_vol.affine, mask_vol.header)
        nib.save(new_img, os.path.join(out_dir, f'sub-{subject}_mask.nii.gz'))

        # save the betaseries image
        out_data = np.zeros([*mask_img.shape, beta.shape[0]])
        out_data[mask_img, :] = beta.T
        new_img = nib.Nifti1Image(out_data, mask_vol.affine, mask_vol.header)
        nib.save(new_img, os.path.join(out_dir, f'sub-{subject}_beta.nii.gz'))

        # save the residuals image
        for run, res in resid.items():
            out_data = np.zeros([*mask_img.shape, res.shape[0]])
            out_data[mask_img, :] = res.T
            new_img = nib.Nifti1Image(out_data, mask_vol.affine, mask_vol.header)
            nib.save(
                new_img, os.path.join(out_dir, f'sub-{subject}_run-{run}_resid.nii.gz')
            )
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
    parser.add_argument(
        '--exclude-motion', '-x', action='store_true', help='exclude motion outliers'
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
        args.exclude_motion,
    )
