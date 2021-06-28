#!/usr/bin/env python
#
# Export behavioral task events to BIDS format.

import os
import argparse
import json
from pkg_resources import resource_filename
import numpy as np

from tesser import tasks


def write_events(data, keys, bids_dir, task, data_type, file_type):
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
            file = os.path.join(subj_dir, f'sub-{subject}_task-{task}_{file_type}.tsv')
        run_data[keys].to_csv(
            file, sep='\t', index=False, na_rep='n/a', float_format='%.3f'
        )


def copy_json(in_file, out_file):
    """Copy a JSON file."""
    with open(in_file, 'r') as f:
        j = json.load(f)

    with open(out_file, 'w') as f:
        json.dump(j, f, indent=4)


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
    data = struct.query('part == 1').copy()
    duration = 1.5
    for (subject, run), run_data in data.groupby(['subject', 'run']):
        include = data.eval(f'subject == {subject} and run == {run}')
        data.loc[include, 'onset'] = np.arange(0, len(run_data) * duration, duration)
        data.loc[include, 'duration'] = duration
    write_events(data, struct_keys, bids_dir, 'learn', 'beh', 'events')
    json_file = resource_filename('tesser', 'data/task-learn_events.json')
    copy_json(json_file, os.path.join(bids_dir, 'task-learn_events.json'))

    # struct
    data = struct.query('part == 2')
    write_events(data, struct_keys, bids_dir, 'struct', 'func', 'events')
    json_file = resource_filename('tesser', 'data/task-struct_events.json')
    copy_json(json_file, os.path.join(bids_dir, 'task-struct_events.json'))

    # induct
    max_duration = 8
    fix_duration = 0.5
    induct = tasks.load_induct(data_dir, subjects).copy()
    induct['run'] = 1
    induct['response'] = induct['response'].astype('Int64')
    for subject in induct['subject'].unique():
        include = induct.eval(f'subject == {subject}')
        duration = induct.loc[include, 'response_time'].fillna(max_duration)
        onset = (duration + fix_duration).cumsum().shift(1).fillna(0)
        induct.loc[include, 'onset'] = onset
        induct.loc[include, 'duration'] = duration
    induct_keys = [
        'onset',
        'duration',
        'trial_type',
        'environment',
        'community',
        'cue',
        'opt1',
        'opt2',
        'within_opt',
        'response',
        'response_time',
    ]
    write_events(induct, induct_keys, bids_dir, 'induct', 'beh', 'events')
    json_file = resource_filename('tesser', 'data/task-induct_events.json')
    copy_json(json_file, os.path.join(bids_dir, 'task-induct_events.json'))

    # parse
    parse = tasks.load_parse(data_dir, subjects).copy()
    parse_keys = [
        'onset',
        'duration',
        'trial_type',
        'path_type',
        'community',
        'object',
        'object_type',
        'response',
        'response_time',
    ]
    duration = 1.5
    for (subject, run), run_data in parse.groupby(['subject', 'run']):
        include = parse.eval(f'subject == {subject} and run == {run}')
        parse.loc[include, 'onset'] = np.arange(0, len(run_data) * duration, duration)
        parse.loc[include, 'duration'] = duration
    write_events(parse, parse_keys, bids_dir, 'parse', 'beh', 'events')
    json_file = resource_filename('tesser', 'data/task-parse_events.json')
    copy_json(json_file, os.path.join(bids_dir, 'task-parse_events.json'))

    # group
    group = tasks.load_group(data_dir, subjects).copy()
    group['run'] = 1
    group_keys = ['object', 'object_type', 'community', 'dim1', 'dim2']
    write_events(group, group_keys, bids_dir, 'group', 'beh', 'beh')
    json_file = resource_filename('tesser', 'data/task-group_beh.json')
    copy_json(json_file, os.path.join(bids_dir, 'task-group_beh.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study_dir', help='main data directory')
    parser.add_argument('bids_dir', help='output directory for BIDS files')
    args = parser.parse_args()
    main(args.study_dir, args.bids_dir)
