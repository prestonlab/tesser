#!/usr/bin/env python
#
# Apply post-processing to a BIDS exported dataset.

import os
import stat
import json
import argparse
from bids import BIDSLayout


def fix_dict(j):
    if 'time' in j and 'samples' in j['time']:
        if 'DataSetTrailingPadding' in j['time']['samples']:
            del j['time']['samples']['DataSetTrailingPadding']

    if 'global' in j:
        if 'slices' in j['global']:
            if 'DataSetTrailingPadding' in j['global']['slices']:
                del j['global']['slices']['DataSetTrailingPadding']
        if 'const' in j['global']:
            if 'DataSetTrailingPadding' in j['global']['const']:
                del j['global']['const']['DataSetTrailingPadding']


def main(data_dir):
    # get all imaging data sidecar files
    prw = stat.S_IWRITE | stat.S_IREAD
    layout = BIDSLayout(data_dir)
    json_files = layout.get(datatype=['anat', 'fmap', 'func'], extension='json')
    for json_file in json_files:
        # load the sidecar into a dictionary
        prop = json_file.get_dict()

        # remove the offending fields
        fix_dict(prop)

        # write a fixed version
        os.chmod(json_file.path, prw)
        with open(json_file.path, 'w') as f:
            json.dump(prop, f, indent=4)

    # add IntendedFor field to fieldmaps
    func_runs = {1: [1, 2, 3], 2: [4, 5, 6]}
    for subject in layout.get_subjects():
        for run in range(1, 3):
            func_files = [
                f'func/sub-{subject}_task-struct_run-{func_run}_bold.nii.gz'
                for func_run in func_runs[run]
            ]
            sbref_files = [
                f'func/sub-{subject}_task-struct_run-{func_run}_sbref.nii.gz'
                for func_run in func_runs[run]
            ]
            fmap_files = layout.get(
                datatype='fmap', extension='json', subject=subject, run=run
            )
            for fmap_file in fmap_files:
                prop = fmap_file.get_dict()
                prop['IntendedFor'] = func_files + sbref_files
                os.chmod(fmap_file.path, prw)
                with open(fmap_file.path, 'w') as f:
                    json.dump(prop, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='path to BIDS dataset')
    args = parser.parse_args()
    main(args.data_dir)
