#!/usr/bin/env python
#
# Estimate a betaseries with each object in each run for an ROI.

import os
import argparse
import numpy as np
from tesser import rsa


def main(raw_dir, post_dir, mask, bold, subject):
    # run betaseries estimation for each run
    beta = np.hstack(
        [
            rsa.run_betaseries(
                raw_dir, post_dir, mask, bold, subject, run
            ) for run in range(1, 7)
        ]
    )

    # save a numpy array with the results
    out_dir = os.path.join(post_dir, 'results', 'beta', bold, mask)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'beta_{subject}.np')
    np.save(out_file, beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='path to raw data BIDS directory')
    parser.add_argument('post_dir', help='path to directory with model results')
    parser.add_argument('mask', help='desc of mask defining the ROI')
    parser.add_argument('bold', help='desc of preprocessed bold images')
    parser.add_argument('subject', help='subject identifier (e.g., 100)')
    args = parser.parse_args()
    main(args.raw_dir, args.post_dir, args.mask, args.bold, args.subject)
