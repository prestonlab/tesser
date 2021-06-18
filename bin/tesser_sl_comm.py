#!/usr/bin/env python
#
# Run a searchlight to search for community structure.
import argparse
import os
import numpy as np
import scipy.spatial.distance as sd
import pandas as pd
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
from mindstorm import prsa


def perm_indices(n_perm, run, n_object):
    """Set indices to permute object."""
    n_run = len(np.unique(run))
    ind = [
        np.tile(
            np.random.permutation(n_object), n_run
        ) + (run - 1) * n_object for _ in range(n_perm - 1)
    ]
    ind.insert(0, np.arange(n_run * n_object))
    return ind


def within_across(subj, mask, sl_rad, var):
    """Run permutation test of within contrasted with across."""
    data = subj[0][mask, :].T
    ind = var['ind']
    n_perm = len(ind)

    sim = 1 - sd.squareform(sd.pdist(data, 'correlation'))
    stat = np.zeros((2, n_perm))
    for i, perm_ind in enumerate(ind):
        sim_perm = sim[np.ix_(perm_ind, perm_ind)]
        stat[0, i] = np.mean(sim_perm[var['within']])
        stat[1, i] = np.mean(sim_perm[var['across']])

    z = (
        prsa.perm_z(stat[0]),
        prsa.perm_z(stat[1]),
        prsa.perm_z(stat[0] - stat[1]),
        prsa.perm_z(stat[1] - stat[0]),
    )
    return z


def main(model_dir, subject, beta, mask, n_perm=1000, n_proc=None):
    beta_dir = os.path.join(model_dir, 'results', 'beta', beta, mask, f'sub-{subject}')
    beta_file = os.path.join(beta_dir, f'sub-{subject}_beta.nii.gz')
    mask_file = os.path.join(beta_dir, f'sub-{subject}_mask.nii.gz')
    events_file = os.path.join(beta_dir, f'sub-{subject}_events.tsv')

    # load beta series events
    events = pd.read_table(events_file)
    run = events['run'].to_numpy()
    community = events['community'].to_numpy()
    objects = events['object'].to_numpy()

    # pair inclusion tests
    across_run = run != run[:, np.newaxis]
    within_community = community == community[:, np.newaxis]
    across_community = community != community[:, np.newaxis]
    diff_object = objects != objects[:, np.newaxis]
    lower = np.tril(np.ones((len(run), len(run)), dtype=bool), -1)

    # bin definition
    include_within = across_run & within_community & diff_object & lower
    include_across = across_run & across_community & diff_object & lower

    # load images
    beta_img = nib.load(beta_file)
    mask_img = nib.load(mask_file)
    beta = beta_img.get_fdata()
    mask = mask_img.get_fdata().astype(bool)

    # run searchlight
    sl = Searchlight(
        sl_rad=3, max_blk_edge=5, shape=Ball, min_active_voxels_proportion=0
    )
    n_object = events['object'].nunique()
    ind = perm_indices(n_perm, run, n_object)
    bcast_var = {
        'ind': ind,
        'within': include_within,
        'across': include_across,
    }
    sl.distribute([beta], mask)
    sl.broadcast(bcast_var)
    outputs = sl.run_searchlight(within_across, pool_size=n_proc)

    # unpack searchlight output
    names = [
        'within',
        'across',
        'withinMinusAcross',
        'acrossMinusWithin',
    ]
    d1, d2, d3 = mask.shape
    zstat = np.zeros((d1, d2, d3, len(names)))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                if outputs[i, j, k] is not None:
                    zstat[i, j, k, :] = outputs[i, j, k]

    # save searchlight results images
    for i, name in enumerate(names):
        new_img = nib.Nifti1Image(zstat[..., i], mask_img.affine, mask_img.header)
        out_file = os.path.join(beta_dir, f'sub-{subject}_desc-{name}_zstat.nii.gz')
        nib.save(new_img, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='path to model director')
    parser.add_argument('subject', help='subject ID')
    parser.add_argument('beta', help='name of betaseries')
    parser.add_argument('mask', help='name of mask')
    parser.add_argument(
        '--n-perm', '-p', type=int, default=1000, help='number of permutations'
    )
    parser.add_argument('--n-proc', '-n', type=int, help='number of processes')
    args = parser.parse_args()
    main(
        args.model_dir,
        args.subject,
        args.beta,
        args.mask,
        n_perm=args.n_perm,
        n_proc=args.n_proc,
    )
