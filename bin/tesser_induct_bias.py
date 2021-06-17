#!/usr/bin/env python
#
# Write out subject induction bias.

import os
import argparse
import numpy as np
from scipy import stats
import pandas as pd
from tesser import tasks


def main(raw_dir, out_file, zscore):
    subjects = tasks.get_subj_list()
    bias = np.zeros(len(subjects))
    for i, subject in enumerate(subjects):
        induct_file = os.path.join(
            raw_dir, f'sub-{subject}', 'beh', f'sub-{subject}_task-induct_events.tsv'
        )
        induct = pd.read_table(induct_file)
        induct['correct'] = induct['within_opt'] == induct['response']
        exclude = induct['response'].isna()
        induct.loc[exclude, 'correct'] = np.nan
        bias[i] = induct['correct'].mean()
    if zscore:
        bias = stats.zscore(bias)
    np.savetxt(out_file, bias)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw bids directory')
    parser.add_argument('out_file', help='path to output text file')
    parser.add_argument(
        '--zscore', '-z', action='store_true', help='zscore bias before saving'
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_file, args.zscore)
