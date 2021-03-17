#!/usr/bin/env python
#
# Run Bayesian representational similarity analysis in a searchlight.

import os
import argparse
from brainiak.reprsimil import brsa
from tesser import mvpa
from tesser import rsa


def main(subject, study_dir, mask, feature_mask, res_dir, radius=3, n_proc=None):
    from mvpa2.measures.searchlight import sphere_searchlight
    from mvpa2.datasets.mri import map2nifti

    # load functional data
    subject_dir = os.path.join(study_dir, f'tesser_{subject}')
    ds = mvpa.load_struct_timeseries(
        study_dir, subject, mask, feature_mask=feature_mask, verbose=1, zscore_run=True
    )

    # load events data, split by object within structure learning block
    vols = rsa.load_vol_info(study_dir, subject)
    events = vols.query('sequence_type > 0').copy()
    n_item = events['trial_type'].nunique()
    events['trial_type'] = (
        events['trial_type'] + (events['sequence_type'] - 1) * n_item
    )

    # set up BRSA model
    model = brsa.GBRSA()
    n_ev = 21 * 2
    n_vol = ds.shape[0]
    mat, nuisance, scan_onsets = rsa.create_brsa_matrix(subject_dir, events, n_vol)
    m = mvpa.ItemBRSA(model, n_ev, mat, nuisance, scan_onsets)

    # run searchlight
    sl = sphere_searchlight(m, radius=radius, nproc=n_proc)
    sl_map = sl(ds)

    # save included voxels map
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    nifti_include = map2nifti(ds, sl_map[-1])
    include_file = os.path.join(res_dir, 'included.nii.gz')
    nifti_include.to_filename(include_file)

    # save pairwise correlations as a timeseries
    filepath = os.path.join(res_dir, 'stat.nii.gz')
    nifti = map2nifti(ds, sl_map[:-1])
    nifti.to_filename(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', help="ID of subject to process.")
    parser.add_argument('mask', help="name of mask for searchlight centers")
    parser.add_argument('feature_mask', help="name of mask for included voxels")
    parser.add_argument('res_dir', help="path to directory to save results.")
    parser.add_argument('--study-dir', help="path to main study data directory.")
    parser.add_argument(
        '--radius', '-r', type=int, default=3, help="searchlight radius"
    )
    parser.add_argument(
        '--n-proc', '-n', type=int, default=None, help="processes for searchlight"
    )
    args = parser.parse_args()

    if args.study_dir is None:
        if 'STUDYDIR' not in os.environ:
            raise RuntimeError('STUDYDIR environment variable not set.')
        env_study_dir = os.environ['STUDYDIR']
    else:
        env_study_dir = args.study_dir
    main(
        args.subject, env_study_dir, args.mask, args.feature_mask, args.res_dir,
        radius=args.radius, n_proc=args.n_proc
    )
