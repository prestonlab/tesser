#!/usr/bin/env python
#
# Extract the betaseries pattern for a cluster.

import os
import argparse
import subprocess as sub
import numpy as np
import nibabel as nib
from tesser import tasks
from tesser import rsa


def main(raw_dir, beta_dir, model_name, cluster_ind, cluster_name):
    if model_name == 'community':
        stat_dir = os.path.join(beta_dir, model_name)
    else:
        stat_dir = os.path.join(beta_dir, f'community_{model_name}')
    cluster_mask = os.path.join(stat_dir, 'cluster_mask10.nii.gz')
    mask = os.path.join(stat_dir, 'mask.nii.gz')
    cluster_dir = os.path.join(stat_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)

    # extract the cluster to make a mask
    roi_mask = os.path.join(cluster_dir, f'desc-{model_name}{cluster_name}')
    print('Extracting cluster...')
    result = sub.run(
        [
            'fslmaths',
            cluster_mask,
            '-thr',
            cluster_ind,
            '-uthr',
            cluster_ind,
            '-bin',
            roi_mask,
        ], capture_output=True, text=True
    )
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # dilate the mask by 1 voxel
    dil_mask = os.path.join(cluster_dir, f'desc-{model_name}{cluster_name}dil1')
    print('Dilating cluster...')
    result = sub.run(
        [
            'fslmaths',
            roi_mask,
            '-kernel',
            'sphere',
            '1.75',
            '-dilD',
            '-mas',
            mask,
            dil_mask
        ], capture_output=True, text=True
    )
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    subjects = tasks.get_subj_list()
    for subject in subjects:
        # load betaseries as a matrix
        bold_file = os.path.join(
            beta_dir, f'sub-{subject}', f'sub-{subject}_beta.nii.gz'
        )
        mask_file = os.path.join(
            beta_dir, f'sub-{subject}', f'sub-{subject}_mask.nii.gz'
        )
        bold_vol = nib.load(bold_file)
        mask_vol = nib.load(mask_file)
        bold_img = bold_vol.get_fdata()
        mask_img = mask_vol.get_fdata().astype(bool)
        beta = bold_img[mask_img].T

        # save to a numpy file
        mat_file = os.path.join(
            cluster_dir, f'sub-{subject}_desc-{model_name}{cluster_name}dil1_beta.npy',
        )
        np.save(mat_file, beta)

        # save events
        vols = rsa.load_struct_vols(raw_dir, subject)
        events = vols.copy()
        events['onset'] = np.arange(len(events))
        events['duration'] = 1.0
        events.to_csv(
            os.path.join(cluster_dir, f'sub-{subject}_events.tsv'), sep='\t',
            index=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw BIDS directory')
    parser.add_argument('beta_dir', help='path to betaseries directory')
    parser.add_argument('model_name', help='name of randomise model')
    parser.add_argument('cluster_ind', help='index of cluster in cluster_mask10.nii.gz')
    parser.add_argument('cluster_name', help='name to give cluster')
    args = parser.parse_args()
    main(
        args.raw_dir,
        args.beta_dir,
        args.model_name,
        args.cluster_ind,
        args.cluster_name,
    )
