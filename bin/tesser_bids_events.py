#!/usr/bin/env python
#
# Export behavioral task events to BIDS format.

import os
import argparse
import shutil
from pkg_resources import resource_filename
import numpy as np

from tesser import tasks


def write_events(data, keys, bids_dir, task, data_type):
    """Write run events."""
    multiple_runs = data['run'].nunique() > 1
    for (subject, run), run_data in data.groupby(['subject', 'run']):
        subj_dir = os.path.join(bids_dir, f'sub-{subject}', data_type)
        os.makedirs(subj_dir, exist_ok=True)
        if multiple_runs:
            file = os.path.join(
                subj_dir, f'sub-{subject}_task-{task}_run-{run}_events.tsv'
            )
        else:
            file = os.path.join(subj_dir, f'sub-{subject}_task-{task}_events.tsv')
        run_data[keys].to_csv(
            file, sep='\t', index=False, na_rep='n/a', float_format='%.3f'
        )


def main(study_dir, bids_dir):
    data_dir = os.path.join(study_dir, 'Data')
    scan_dir = os.path.join(study_dir, 'TesserScan')
    subjects = tasks.get_subj_list()

    # structure task
    struct_onsets = tasks.load_struct_onsets(scan_dir, subjects)
    struct = tasks.load_struct(data_dir, subjects, struct_onsets)
    struct_keys = [
        'onset',
        'duration',
        'trial_type',
        'block',
        'community',
        'object',
        'object_type',
        'orientation',
        'response',
        'response_time',
    ]

    # learn
    data = struct.query('part == 1')[struct_keys].copy()
    duration = 1.5
    for (subject, run), _ in data.groupby(['subject', 'run']):
        include = data.eval(f'subject == {subject} and run == {run}')
        data.loc[include, 'onset'] = np.arange(0, len(data) * duration, duration)
        data.loc[include, 'duration'] = duration
    write_events(data, bids_dir, 'learn', 'beh')
    json_file = resource_filename('tesser', 'data/task-learn_events.json')
    shutil.copy(json_file, os.path.join(bids_dir, 'task-learn_events.json'))

    # struct
    data = struct.query('part == 2')
    write_events(data, bids_dir, 'struct', 'beh')
    json_file = resource_filename('tesser', 'data/task-struct_events.json')
    shutil.copy(json_file, os.path.join(bids_dir, 'task-struct_events.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study_dir', help='main data directory')
    parser.add_argument('bids_dir', help='output directory for BIDS files')
    args = parser.parse_args()
    main(args.study_dir, args.bids_dir)
