"""Utilities for loading Tesser behavioral data."""

import numpy as np
import pandas as pd
import os
from glob import glob
from tesser import network


def get_subj_list():
    """Get IDs of included tesser participants."""
    participant_list = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 135, 136, 137, 138
    ]
    return participant_list


def get_subj_dir(data_dir, subject_num):
    """
    Get the path to the data directory for a subject.

    Parameters
    ----------
    data_dir : str
        Path to the base directory with subdirectories for each
        subject.

    subject_num : int
        Subject number without initials.

    Returns
    -------
    subj_dir : str
        Path to the base directory for the subject.
    """
    # check that the base directory exists
    if not os.path.exists(data_dir):
        raise IOError(f'Directory does not exist: {data_dir}')

    # look for directories with the correct pattern
    dir_search = glob(os.path.join(data_dir, f'tesserScan_{subject_num}_*'))
    if len(dir_search) != 1:
        raise IOError(f'Problem finding subject directory for {subject_num}')
    return dir_search[0]


def load_phase_subject(data_dir, subject_num, file_pattern):
    """Load log as a DataFrame."""
    # get the first file matching the log file pattern
    subj_dir = get_subj_dir(data_dir, subject_num)
    file_search = glob(os.path.join(subj_dir, file_pattern))
    if len(file_search) != 1:
        raise IOError(f'Problem finding log: {file_pattern}')
    run_file = file_search[0]

    # read log, fixing problem with spaces in column names
    df = pd.read_csv(run_file, sep='\t', skipinitialspace=True)
    return df


def load_struct_run(data_dir, subject_num, part_num, run_num):
    """Load dataframe for one structured learning run."""
    # load structure learning run
    file_pattern = (
        f'tesserScan_{subject_num}_*_StructLearn_Part{part_num}_Run_{run_num}.txt'
    )
    df = load_phase_subject(data_dir, subject_num, file_pattern)

    # add a field indicating the experiment part
    df['part'] = part_num

    # remove the fixation trials in the runs (these are just filler trials)
    df = df[pd.notnull(df['objnum'])]

    # convert object labels to integer
    df = df.astype({'objnum': 'int'})
    return df


def load_struct_subject(data_dir, subject_num):
    """Load dataframe with structured learning task for one subject."""
    # list of all runs to load
    parts = (1, 2)
    part_runs = {1: range(1, 6), 2: range(1, 7)}

    # load individual runs
    df_list = []
    for part in parts:
        for run in part_runs[part]:
            run_df = load_struct_run(data_dir, subject_num, part, run)
            df_list.append(run_df)

    # concatenate into one data frame
    df = pd.concat(df_list, sort=False, ignore_index=True)
    return df


def load_struct(data_dir, subjects=None):
    """Load structure learning data for multiple subjects."""
    if subjects is None:
        subjects = get_subj_list()

    # load raw data for all subjects
    df_all = []
    for subject in subjects:
        df_subj = load_struct_subject(data_dir, subject)
        df_all.append(df_subj)
    raw = pd.concat(df_all, axis=0, ignore_index=True)

    # community info
    nodes = network.node_info()
    raw_nodes = nodes.loc[raw['objnum'], :].reset_index()

    # convert to BIDS format
    orientation = {'cor': 'canonical', 'rot': 'rotated'}
    response = {'c': 'canonical', 'n': 'rotated'}
    object_type = {0: 'central', 1: 'boundary'}
    df = pd.DataFrame(
        {
            'subject': raw['SubjNum'],
            'part': raw['part'],
            'run': raw['run'],
            'trial': raw['trial'],
            'trial_type': raw['seqtype'].astype('Int64'),
            'community': raw_nodes['community'],
            'object': raw['objnum'],
            'object_type': raw_nodes['node_type'].map(object_type).astype('category'),
            'orientation': raw['orientnam'].map(orientation).astype('category'),
            'response': raw['resp'].map(response).astype('category'),
            'response_time': raw['rt'],
            'correct': raw['acc'].astype('Int64'),
        }
    )
    return df


def load_induct_subject(data_dir, subject_num):
    """Load dataframe of inductive generalization task for one subject."""
    file_pattern = f'tesserScan_{subject_num}_*_InductGen.txt'
    df = load_phase_subject(data_dir, subject_num, file_pattern)
    return df


def load_induct(data_dir, subjects=None):
    """Load induction data in BIDs format."""
    if subjects is None:
        subjects = get_subj_list()

    # load subject data
    df_all = []
    for subject in subjects:
        df_subj = load_induct_subject(data_dir, subject)
        df_all.append(df_subj)
    raw = pd.concat(df_all, axis=0, ignore_index=True)

    # add node information
    nodes = network.node_info()

    # convert to BIDS format
    trial_type = {'Prim': 'central', 'Bound1': 'boundary1', 'Bound2': 'boundary2'}
    df = pd.DataFrame(
        {
            'subject': raw['SubjNum'],
            'trial': raw['TrialNum'],
            'trial_type': raw['QuestType'].map(trial_type).astype('category'),
            'environment': raw['Environment'],
            'community': nodes.loc[raw['CueNum'], 'community'].to_numpy(),
            'cue': raw['CueNum'],
            'opt1': raw['Opt1Num'],
            'opt2': raw['Opt2Num'],
            'response': raw['Resp'].astype('Int64'),
            'response_time': raw['RT'],
            'correct': raw['Acc'].astype('Int64'),
        }
    )
    df['trial_type'].cat.reorder_categories(
        ['central', 'boundary1', 'boundary2'], inplace=True
    )
    return df


def load_parse_subject(data_dir, subject_num):
    """Load parsing data for one subject."""
    # search for a file with the correct name formatting
    file_pattern = f'tesserScan_{subject_num}_*_StructParse.txt'
    df = load_phase_subject(data_dir, subject_num, file_pattern)
    return df


def load_group_subject(data_dir, subject_num):
    """Load matrix of grouping data."""
    # subject directory
    subj_dir = get_subj_dir(data_dir, subject_num)

    # search for a file with the correct name formatting
    file_pattern = f'{subject_num}_FinalGraph.txt'
    file_search = glob(os.path.join(subj_dir, file_pattern))
    if len(file_search) != 1:
        raise IOError(f'Problem finding log: {file_pattern}')

    # read log, fixing problem with spaces in column names
    mat = np.loadtxt(file_search[0]).astype(int)
    return mat


def load_parse(data_dir, subjects=None):
    """Load induction data in BIDs format."""
    if subjects is None:
        subjects = get_subj_list()

    # load subject data
    df_all = []
    for subject in subjects:
        df_subj = load_parse_subject(data_dir, subject)
        df_all.append(df_subj)
    raw = pd.concat(df_all, axis=0, ignore_index=True)

    # add node information
    nodes = network.node_info()
    raw_nodes = nodes.loc[raw['objnum'], :].reset_index()

    # convert to BIDS format
    trial_type = {1: 'random', 2: 'forward', 3: 'backward'}
    response_type = {'PARSED': 1, 'NONE': 0}
    object_type = {0: 'central', 1: 'boundary'}
    df = pd.DataFrame(
        {
            'subject': raw['SubjNum'],
            'run': raw['run'],
            'trial': raw['trial'],
            'trial_type': raw['objseq'].map(trial_type).astype('category'),
            'community': raw_nodes['community'],
            'object': raw['objnum'],
            'object_type': raw_nodes['node_type'].map(object_type).astype('category'),
            'response': raw['resp'].map(response_type).astype('Int64'),
            'response_time': raw['rt'],
        }
    )
    return df
