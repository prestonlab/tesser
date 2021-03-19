#!/usr/bin/env python
#
# Run Bayesian RSA on a tesser participant.

import os
import argparse
import numpy as np
from nilearn import input_data
from brainiak.reprsimil import brsa
from tesser import rsa


def main(study_dir, subject, roi, res_dir, blocks='combined'):
    # load task information
    vols = rsa.load_vol_info(study_dir, subject)

    events = vols.query('sequence_type > 0').copy()
    if blocks == 'separate':
        # separately model trials in walk and random blocks
        n_item = events['trial_type'].nunique()
        events['trial_type'] = (
            events['trial_type'] + (events['sequence_type'] - 1) * n_item
        )
    elif blocks != 'combined':
        raise ValueError(f'Invalid blocks option: {blocks}')

    # get mask image
    subject_dir = os.path.join(study_dir, f'tesser_{subject}')
    mask_image = os.path.join(
        subject_dir, 'anatomy', 'antsreg', 'data', 'funcunwarpspace',
        'rois', 'mni', f'{roi}.nii.gz'
    )

    # load masked functional images
    runs = list(range(1, 7))
    bold_images = [
        os.path.join(
            subject_dir, 'BOLD', 'antsreg', 'data',
            f'functional_run_{run}_bold_mcf_brain_corr_notemp.feat',
            'filtered_func_data.nii.gz'
        ) for run in runs
    ]
    masker = input_data.NiftiMasker(mask_img=mask_image)
    image = np.vstack(
        [masker.fit_transform(bold_image) for bold_image in bold_images]
    )

    # create design matrix
    n_vol = image.shape[0]
    mat, nuisance, scan_onsets = rsa.create_brsa_matrix(subject_dir, events, n_vol)

    # run Bayesian RSA
    scan_onsets = np.arange(0, image.shape[0], image.shape[0] / len(runs))
    model = brsa.GBRSA()
    model.fit([image], [mat], nuisance=nuisance, scan_onsets=scan_onsets)

    # save results
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    var_names = [
        'U', 'L', 'C', 'nSNR', 'sigma', 'rho', 'beta', 'beta0',
        'X0', 'beta0_null', 'X0_null', 'n_nureg'
    ]
    results = {var: getattr(model, var + '_') for var in var_names}
    out_file = os.path.join(res_dir, f'sub-{subject}_brsa.npz')
    np.savez(out_file, **results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help="ID of subject to process.")
    parser.add_argument('roi', help="name of mask to use.")
    parser.add_argument('res_dir', help="path to directory to save results.")
    parser.add_argument('--study-dir', help="path to main study data directory.")
    parser.add_argument(
        '--blocks', '-b',
        help="blocks to include in model ['walk'|'random'|'combined'|'separate']"
    )
    args = parser.parse_args()

    if args.study_dir is None:
        if 'STUDYDIR' not in os.environ:
            raise RuntimeError('STUDYDIR environment variable not set.')
        env_study_dir = os.environ['STUDYDIR']
    else:
        env_study_dir = args.study_dir

    main(env_study_dir, args.subject, args.roi, args.res_dir, blocks=args.blocks)
