#!/usr/bin/env python
#
# Export behavioral task events to BIDS format.

import os
import argparse
import shutil
from pkg_resources import resource_filename

from tesser import tasks


def main(study_dir, bids_dir):
    data_dir = os.path.join(study_dir, 'Data')
    scan_dir = os.path.join(study_dir, 'TesserScan')
    subjects = tasks.get_subj_list()
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
    struct = struct.query('part == 2')
    for (subject, run), data in struct.groupby(['subject', 'run']):
        subj_dir = os.path.join(bids_dir, f'sub-{subject}', 'func')
        os.makedirs(subj_dir, exist_ok=True)
        data = data[struct_keys]
        file = os.path.join(subj_dir, f'sub-{subject}_task-struct_run-{run}_events')
        data.to_csv(file + '.tsv', sep='\t', index=False, na_rep='n/a')
    json_file = resource_filename('tesser', 'data/task-struct_events.json')
    shutil.copy(json_file, os.path.join(bids_dir, 'task-struct_events.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study_dir', help='main data directory')
    parser.add_argument('bids_dir', help='output directory for BIDS files')
    args = parser.parse_args()
    main(args.study_dir, args.bids_dir)
