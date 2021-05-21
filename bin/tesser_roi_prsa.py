#!/usr/bin/env python
#
# Run partial representational similarity analysis on a region.

import os
import warnings
import logging
import numpy as np
import scipy.spatial.distance as sd
import pandas as pd

from mindstorm import subjutil
from mindstorm import prsa
from tesser import tasks
from tesser import model

warnings.simplefilter('ignore', FutureWarning)
from tesser import rsa


def main(
    subject,
    study_dir,
    beh_dir,
    rsa_name,
    roi,
    res_name,
    block=None,
    n_perm=1000,
    invert=False,
):
    # set up log
    res_dir = os.path.join(study_dir, 'batch', 'prsa', res_name, roi)
    log_dir = os.path.join(res_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_sub-{subject}.txt')
    logging.basicConfig(
        filename=log_file, filemode='w', level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
    )
    logging.info(f'Analyzing data from subject {subject} and ROI {roi}.')

    # load dissimilarity matrix
    rsa_dir = os.path.join(study_dir, 'batch', 'rsa', rsa_name, roi)
    logging.info(f'Loading BRSA correlations from {rsa_dir}.')
    roi_rdm = 1 - np.load(os.path.join(rsa_dir, f'sub-{subject}_brsa.npz'))['C']

    # get trial pairs to test
    n_state = 21
    if block is not None and roi_rdm.shape[0] > n_state:
        if block == 'walk':
            roi_rdm = roi_rdm[:n_state, :n_state]
        elif block == 'random':
            roi_rdm = roi_rdm[n_state:, n_state:]
        else:
            raise ValueError(f'Invalid block type: {block}')
        logging.info(f'Analyzing the {block} blocks only.')

    # load structure learning data
    logging.info(f'Loading behavioral data from {beh_dir}.')
    struct = tasks.load_struct(beh_dir, [subject])

    # simple model: items in different communities are less similar
    comm = struct.groupby('object')['community'].first().to_numpy()
    comm_rdm = (comm[:, None] != comm).astype(float)

    # learning model based on part 1
    # dissimilarity is inversely proportionate to association strength
    struct1 = struct.query('part == 1').copy()
    gamma = .97424
    alpha = 0.1
    sr_mat = model.learn_struct_sr(struct1, gamma, alpha, n_state)
    sr_rdm = rsa.make_sym_matrix(1 - sr_mat / np.sum(sr_mat))

    # create model set
    if invert:
        logging.info('Using inverted models.')
        comm_rdm = 1 - comm_rdm
        comm_rdm[np.arange(n_state), np.arange(n_state)] = 0
        sr_rdm = 1 - sr_rdm
        sr_rdm[np.arange(n_state), np.arange(n_state)] = 0
    model_rdms = [comm_rdm, sr_rdm]
    model_names = ['community', 'sr']

    # initialize the permutation test
    logging.info('Initializing PRSA test.')
    perm = prsa.init_pRSA(n_perm, model_rdms, rank=False)
    data_vec = sd.pdist(roi_rdm, 'correlation')
    n_model = len(model_rdms)

    # calculate permutation correlations
    logging.info('Testing significance of partial correlations.')
    rho = np.zeros(n_model)
    zstat = np.zeros(n_model)
    for i in range(n_model):
        stat = prsa.perm_partial(
            data_vec, perm['model_mats'][i], perm['model_resid'][i]
        )
        rho[i] = stat[0]
        zstat[i] = prsa.perm_z(stat)

    # save results
    df = pd.DataFrame({'rho': rho, 'zstat': zstat}, index=model_names)
    res_file = os.path.join(res_dir, f'zstat_{subject}.csv')
    logging.info(f'Saving results to {res_file}.')
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
    parser.add_argument(
        '--invert', '-i', action="store_true", help="use model similarity"
    )
    args = parser.parse_args()
    main(
        args.subject,
        args.study_dir,
        args.beh_dir,
        args.rsa_name,
        args.roi,
        args.res_name,
        block=args.block,
        n_perm=args.n_perm,
        invert=args.invert,
    )
