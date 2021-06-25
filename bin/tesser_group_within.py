#!/usr/bin/env python
#
# Write out subject grouping task within-community distance.

import os
import argparse
import numpy as np
from scipy import stats
import scipy.spatial.distance as sd
import pandas as pd
from tesser import tasks


def main(raw_dir, out_file, zscore):
    subjects = tasks.get_subj_list()
    stat = np.zeros(len(subjects))
    for i, subject in enumerate(subjects):
        # distance between object pairs in the final array
        group_file = os.path.join(
            raw_dir, f'sub-{subject}', 'beh', f'sub-{subject}_task-group_beh.tsv'
        )
        group = pd.read_table(group_file)
        coords = group.filter(like='dim').to_numpy()
        distance_vec = sd.pdist(coords, 'euclidean')

        # within-community comparisons
        community = group['community'].to_numpy()
        within_mat = community == community[:, np.newaxis]
        within_vec = sd.squareform(within_mat, checks=False)

        # negative mean distance within community
        stat[i] = -np.mean(distance_vec[within_vec])
    if zscore:
        stat = stats.zscore(stat)
    np.savetxt(out_file, stat, fmt='%.8f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw bids directory')
    parser.add_argument('out_file', help='path to output text file')
    parser.add_argument(
        '--zscore', '-z', action='store_true', help='zscore bias before saving'
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_file, args.zscore)
