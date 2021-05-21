#!/usr/bin/env python
#
# Print commands to run partial representational similarity analysis.

import os
import argparse
from tesser import tasks
from tesser import rsa


def submit_brsa(subjects, rois, blocks, study_dir, rsa_name):
    if subjects is None:
        subjects = tasks.get_subj_list()

    block_name = {'walk': 'walk', 'random': 'rand'}
    beh_dir = os.path.join(study_dir, 'batch', 'behav')
    for roi in rois:
        inputs = f'{beh_dir} {rsa_name} {roi}'
        for block in blocks:
            options = f'--study-dir={study_dir} -b {block} -p 100000'
            res_name = f'{rsa_name}_{block_name[block]}_com-sr'
            for subject in subjects:
                print(f'tesser_roi_prsa.py {subject} {inputs} {res_name} {options}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rois', help="comma-separated list of masks to use.")
    parser.add_argument('blocks', help="comma-separated list of blocks to analyze.")
    parser.add_argument('rsa_name', help="name of BRSA model to use.")
    parser.add_argument('--study-dir', help="path to main study data directory.")
    parser.add_argument('--subjects', '-s', help="IDs of subjects to process.")
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

    inc_blocks = args.blocks.split(',')
    inc_rois = rsa.parse_rois(args.rois)
    submit_brsa(inc_subjects, inc_rois, inc_blocks, env_study_dir, args.rsa_name)
