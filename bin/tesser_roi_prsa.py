#!/usr/bin/env python
#
# Run partial representational similarity analysis on a region.

import os
import numpy as np
import scipy.spatial.distance as sd
import pandas as pd

from mindstorm import subjutil
from mindstorm import prsa
from tesser import util
from tesser import model
from tesser import rsa


def main(
    subject, study_dir, beh_dir, rsa_name, roi, res_name, block=None, n_perm=1000
):
    # load dissimilarity matrix
    rsa_dir = os.path.join(study_dir, 'batch', 'rsa', rsa_name, roi)
    roi_rdm = 1 - np.load(os.path.join(rsa_dir, f'sub-{subject}_brsa.npz'))['C']

    # get trial pairs to test
    n_state = 21
    if block is not None:
        if block == 'walk':
            roi_rdm = roi_rdm[:n_state, :n_state]
        elif block == 'random':
            roi_rdm = roi_rdm[n_state:, n_state:]
        else:
            raise ValueError(f'Invalid block type: {block}')

    # load structure learning data
    struct = util.load_struct(beh_dir, [subject])

    # simple model: items in different communities are less similar
    comm = struct.groupby('object')['community'].first().to_numpy()
    comm_rdm = (comm[:, None] != comm).astype(float)

    # learning models based on part 1
    # dissimilarity is inversely proportionate to association strength
    struct1 = struct.query('part == 1').copy()
    gamma = [0, .9]
    alpha = 0.05
    sr_mat = [model.learn_sr(struct1, g, alpha, n_state) for g in gamma]
    sr_rdm = [rsa.make_sym_matrix(1 - sr / np.sum(sr)) for sr in sr_mat]

    # create model set
    model_rdms = [comm_rdm] + sr_rdm
    model_names = ['community', 'sr0', 'sr90']

    # initialize the permutation test
    perm = prsa.init_pRSA(n_perm, model_rdms, rank=False)
    data_vec = sd.pdist(roi_rdm)
    n_model = len(model_rdms)

    # calculate permutation correlations
    rho = np.zeros(n_model)
    zstat = np.zeros(n_model)
    for i in range(n_model):
        stat = prsa.perm_partial(
            data_vec, perm['model_mats'][i], perm['model_resid'][i]
        )
        rho[i] = stat[0]
        zstat[i] = prsa.perm_z(stat)

    # save results
    res_dir = os.path.join(study_dir, 'batch', 'prsa', res_name, roi)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    df = pd.DataFrame({'rho': rho, 'zstat': zstat}, index=model_names)
    res_file = os.path.join(res_dir, f'zstat_{subject}.csv')
    df.to_csv(res_file)


if __name__ == '__main__':
    parser = subjutil.SubjParser(include_log=False)
    parser.add_argument('beh_dir', help='path to behavioral data directory')
    parser.add_argument('rsa_name', help='name for rsa results')
    parser.add_argument('roi', help='name of roi to analyze')
    parser.add_argument('res_name', help='name for results')
    parser.add_argument('--block', '-b', help='block to include (walk, random)')
    parser.add_argument(
        '--n-perm', '-p', type=int, default=1000,
        help="number of permutations to run (1000)"
    )
    args = parser.parse_args()
    main(
        args.subject, args.study_dir, args.beh_dir, args.rsa_name, args.roi,
        args.res_name, block=args.block, n_perm=args.n_perm
    )
