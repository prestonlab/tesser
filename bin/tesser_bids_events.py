#!/usr/bin/env python
#
# Export behavioral task events to BIDS format.

import os
import argparse
import json

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

    meta = {
        "onset": {
            "LongName": "Stimulus onset",
            "Description": "Time the object appeared on screen",
            "Units": "s",
        },
        "duration": {
            "LongName": "Stimulus duration",
            "Description": "Time the object remained on screen",
            "Units": "s",
        },
        "trial_type": {
            "LongName": "Block type",
            "Description": "Type of object sequence block",
            "Levels": {
                "structured": "A random walk drawn from the temporal community graph",
                "scrambled": "All objects in random order",
            }
        },
        "block": {
            "LongName": "Block number",
            "Description": "Index of the current block",
        },
        "community": {
            "LongName": "Community number",
            "Description": "The community of the current object",
        },
        "object": {
            "LongName": "Object number",
            "Description": "The node of the current object",
        },
        "object_type": {
            "LongName": "Object type",
            "Description": "Node type of the object in the graph",
            "Levels": {
                "central": "A central node within a community",
                "boundary": "A node connecting to another community",
            }
        },
        "orientation": {
            "LongName": "Object orientation",
            "Description": "Presented orientation of the object",
            "Levels": {
                "canonical": "Same orientation as learned earlier",
                "rotated": "Rotated 90 degrees from original",
            }
        },
        "response": {
            "LongName": "Response",
            "Description": "Response to the rotation detection task",
            "Levels": {
                "canonical": "Same orientation as learned earlier",
                "rotated": "Rotated 90 degrees from original",
            }
        },
        "response_time": {
            "LongName": "Response time",
            "Description": "Time of response to rotation decision",
            "Units": "s",
        },
    }
    file = os.path.join(bids_dir, 'task-struct.json')
    with open(file, 'w') as f:
        json.dump(meta, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study_dir', help='main data directory')
    parser.add_argument('bids_dir', help='output directory for BIDS files')
    args = parser.parse_args()
    main(args.study_dir, args.bids_dir)
