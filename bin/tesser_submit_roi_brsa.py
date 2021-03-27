#!/usr/bin/env python
#
# Submit Bayesian representational similarity analysis.

import os
import argparse
from tesser import util
from tesser import rsa


def submit_brsa(subjects, rois, study_dir, res_dir, blocks):
    if subjects is None:
        subjects = util.get_subj_list()

    options = f'--study-dir={study_dir} -b'
    for roi in rois:
        roi_dir = os.path.join(res_dir, roi)
        for subject in subjects:
            print(f'tesser_roi_brsa.py {subject} {roi} {blocks} {roi_dir} {options}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rois', help="name of mask to use.")
    parser.add_argument(
        'blocks', help="blocks to include ['both','walk','random']", default='both'
    )
    parser.add_argument('res_dir', help="path to directory to save results.")
    parser.add_argument('--study-dir', help="path to main study data directory.")
    parser.add_argument('--subjects', '-s', help="ID of subjects to process.")
    args = parser.parse_args()

    if args.study_dir is None:
        if 'STUDYDIR' not in os.environ:
            raise RuntimeError('STUDYDIR environment variable not set.')
        env_study_dir = os.environ['STUDYDIR']
    else:
        env_study_dir = args.study_dir

    if args.subjects is not None:
        inc_subjects = args.subjects.split(',')
    else:
        inc_subjects = None

    inc_rois = rsa.parse_rois(args.rois)
    submit_brsa(inc_subjects, inc_rois, env_study_dir, args.res_dir, args.blocks)
