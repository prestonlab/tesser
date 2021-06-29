#!/usr/bin/env python
#
# Write out subject induction bias.

import os
import argparse
import numpy as np
from scipy import stats
import pandas as pd
from tesser import tasks


def main(raw_dir, out_file, zscore, contrast=None):
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

        if contrast is not None:
            bias_trial = induct.groupby('trial_type')['correct'].mean()
            if contrast == 'b1b2':
                bias[i] = bias_trial['boundary1'] - bias_trial['boundary2']
            elif contrast == 'b2b1':
                bias[i] = bias_trial['boundary2'] - bias_trial['boundary1']
            else:
                raise ValueError(f'Invalid contrast: {contrast}')
        else:
            bias[i] = induct['correct'].mean()
    if zscore:
        bias = stats.zscore(bias)
    np.savetxt(out_file, bias, fmt='%.8f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw bids directory')
    parser.add_argument('out_file', help='path to output text file')
    parser.add_argument(
        '--zscore', '-z', action='store_true', help='zscore bias before saving'
    )
    parser.add_argument('--contrast', '-c', help='contrast name ("b1b2",)')
    args = parser.parse_args()
    main(args.raw_dir, args.out_file, args.zscore, args.contrast)
