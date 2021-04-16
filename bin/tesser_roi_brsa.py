#!/usr/bin/env python
#
# Run Bayesian RSA on a tesser participant.

import os
import argparse
import warnings
import logging
import numpy as np
from scipy import stats
import nibabel as nib
from brainiak.reprsimil import brsa

warnings.simplefilter('ignore', FutureWarning)
from tesser import rsa


def main(study_dir, subject, roi, res_dir, blocks, method='GBRSA'):
    # set up log
    log_dir = os.path.join(res_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_sub-{subject}.txt')
    logging.basicConfig(
        filename=log_file, filemode='w', level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
    )
    logging.info(f'Analyzing data from subject {subject} and ROI {roi} using {method}.')

    # load task information
    vols = rsa.load_vol_info(study_dir, subject)

    # load events data, split by object within structure learning block
    events = vols.query('sequence_type > 0').copy()
    n_item = events['trial_type'].nunique()
    events['trial_type'] = (
        events['trial_type'] + (events['sequence_type'] - 1) * n_item
    )

    # get mask image
    subject_dir = os.path.join(study_dir, f'tesser_{subject}')
    mask_image = os.path.join(
        subject_dir, 'anatomy', 'antsreg', 'data', 'funcunwarpspace',
        'rois', 'mni', f'{roi}.nii.gz'
    )
    logging.info(f'Masking with {mask_image}.')

    # load mask and get coordinates
    mask_img = nib.load(mask_image)
    mask_data = mask_img.get_fdata()
    mask_ind = np.nonzero(mask_data)
    mask_coords = np.vstack(mask_ind).T

    # get functional image files
    runs = list(range(1, 7))
    bold_images = [
        os.path.join(
            subject_dir, 'BOLD', 'antsreg', 'data',
            f'functional_run_{run}_bold_mcf_brain_corr_notemp.feat',
            'filtered_func_data.nii.gz'
        ) for run in runs
    ]

    # load functional data
    image_list = []
    for bold_image in bold_images:
        logging.info(f'Loading functional data for {bold_image}.')
        bold_img = nib.load(bold_image)
        bold_data = bold_img.get_fdata()
        bold_mat = bold_data[mask_ind].T
        image_list.append(bold_mat)
    image = np.vstack(image_list)
    inten = np.mean(image, 0)
    image = stats.zscore(image, axis=0)

    # create design matrix
    n_vol = image.shape[0]
    mat, nuisance, scan_onsets = rsa.create_brsa_matrix(
        subject_dir, events, n_vol, high_pass=0.003, censor=True, baseline=False
    )

    # split design to get blocks of interest
    if blocks == 'walk':
        include = slice(None, n_item)
        exclude = slice(n_item, None)
    elif blocks == 'random':
        include = slice(n_item, None)
        exclude = slice(None, n_item)
    elif blocks == 'both':
        include = None
        exclude = None
    else:
        raise ValueError(f'Invalid blocks: {blocks}')

    # trim mat to get blocks of interest, add excluded blocks to nuisance
    logging.info(f'Modeling covariance of {blocks} blocks.')
    if exclude is not None:
        mat_include = mat[:, include].copy()
        mat_exclude = mat[:, exclude].copy()
        mat = mat_include
        nuisance = np.hstack((mat_exclude, nuisance))

    # run Bayesian RSA
    n_ev = mat.shape[1]
    if method == 'GBRSA':
        model = brsa.GBRSA(rank=n_ev)
        images = [image]
        mats = [mat]
        kwargs = {}
    elif method == 'BRSA':
        model = brsa.BRSA(rank=n_ev, GP_space=True, GP_inten=True)
        images = image
        mats = mat
        kwargs = {'coords': mask_coords, 'inten': inten}
    else:
        raise ValueError(f'Invalid method: {method}.')

    logging.info(f'Fitting {method} model with rank {n_ev}.')
    try:
        model.fit(images, mats, nuisance=nuisance, scan_onsets=scan_onsets, **kwargs)
    except ValueError:
        logging.exception('Exception during model fitting.')

    # save results
    var_names = [
        'U', 'L', 'C', 'nSNR', 'sigma', 'rho', 'beta', 'beta0',
        'X0', 'beta0_null', 'X0_null', 'n_nureg'
    ]
    results = {var: getattr(model, var + '_') for var in var_names}
    out_file = os.path.join(res_dir, f'sub-{subject}_brsa.npz')
    logging.info(f'Saving results to {out_file}.')
    np.savez(out_file, **results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help="ID of subject to process.")
    parser.add_argument('roi', help="name of mask to use.")
    parser.add_argument(
        'blocks', help="blocks to include ('both','walk','random')",
    )
    parser.add_argument('res_dir', help="path to directory to save results.")
    parser.add_argument('--study-dir', help="path to main study data directory.")
    parser.add_argument(
        '--method', '-m', default='GBRSA', help="modeling method ('BRSA', ['GBRSA'])"
    )
    args = parser.parse_args()

    if args.study_dir is None:
        if 'STUDYDIR' not in os.environ:
            raise RuntimeError('STUDYDIR environment variable not set.')
        env_study_dir = os.environ['STUDYDIR']
    else:
        env_study_dir = args.study_dir

    main(
        env_study_dir, args.subject, args.roi, args.res_dir, args.blocks,
        method=args.method
    )
