#!/usr/bin/env python
#
# Write out average parsing performance.

import argparse
import numpy as np
from scipy import stats
from tesser import tasks


def main(raw_dir, out_file, zscore):
    subjects = tasks.get_subj_list()
    perf = np.zeros(len(subjects))
    for i, subject in enumerate(subjects):
        # get transition labels and length of previous walk
        parse = tasks.load_bids_parse(raw_dir, subject)
        parse = tasks.score_parse(parse)

        # exclude the first walk
        parse = parse.loc[parse['prev_walk'].notna()]

        # contrast community transition and other
        trans_parse = (
            parse.query('transition and prev_walk >= 4')
            .groupby('trial_type')['response']
            .mean()
        )
        other_parse = (
            parse.query('~(transition and prev_walk >= 4)')
            .groupby('trial_type')['response']
            .mean()
        )
        perf_trial_type = trans_parse - other_parse
        perf[i] = perf_trial_type.mean()
    if zscore:
        perf = stats.zscore(perf)
    np.savetxt(out_file, perf, fmt='%.8f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw bids directory')
    parser.add_argument('out_file', help='path to output text file')
    parser.add_argument(
        '--zscore', '-z', action='store_true', help='zscore bias before saving'
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_file, args.zscore)
