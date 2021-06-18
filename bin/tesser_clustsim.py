#!/usr/bin/env python
#
# Estimate null cluster sizes based on average ACF smoothness parameters.

import os
import shutil
import subprocess as sub
import argparse
import numpy as np
import pandas as pd
from tesser import tasks


def main(beta_dir):
    # get average smoothness parameters
    subjects = tasks.get_subj_list()
    all_acf = []
    for subject in subjects:
        subj_dir = os.path.join(beta_dir, f'sub-{subject}')
        for run in range(1, 7):
            smoothness = pd.read_table(
                os.path.join(subj_dir, f'sub-{subject}_run-{run}_smoothness.tsv')
            )
            run_acf = smoothness.to_numpy()[1, :3]
            all_acf.append(run_acf)
    acf = np.mean(np.array(all_acf), 0)
    mask_file = os.path.join(f'sub-{subjects[0]}', f'sub-{subjects[0]}_mask.nii.gz')

    # run 3dClustSim
    out_dir = os.path.join(beta_dir, 'clustsim')
    mask_copy = os.path.join(out_dir, 'mask.nii.gz')
    shutil.copyfile(mask_file, mask_copy)
    acf_str = f'{acf[0]} {acf[1]} {acf[2]}'
    prefix = os.path.join(out_dir, 'clustsim')
    command = (
        f'3dClustSim -mask {mask_copy} -acf {acf_str} -iter 2000 -nodec -prefix {prefix}'
    )
    output = sub.run(command, shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
    print(output.stdout)
    print(output.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('beta_dir', help='path to main beta series directory')
    args = parser.parse_args()
    main(args.beta_dir)
