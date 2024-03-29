"""Work with raw behavioral data."""
import os
from glob import glob

import numpy as np
import pandas as pd
from tesser import network
from tesser.tasks import get_subj_list


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
            if part == 1:
                run_df['block'] = 1
            else:
                run_df['block'] = np.repeat(np.arange(1, 8), 21)
            df_list.append(run_df)

    # concatenate into one data frame
    df = pd.concat(df_list, sort=False, ignore_index=True)
    return df


def load_struct_onsets(scan_dir, subjects):
    """Load onset information for structure scans."""
    columns = ['trial', 'onset', 'tr', 'sequence_type', 'trial_type', 'duration']
    runs = list(range(1, 7))
    subj_list = []
    for subject in subjects:
        data_list = []
        for i, run in enumerate(runs):
            vol_file = os.path.join(
                scan_dir,
                'rsa_allevents_info',
                f'tesser_{subject}_run{run}_info.txt',
            )
            run_data = pd.read_csv(vol_file, names=columns)
            run_data['duration'] = 1
            run_data['run'] = run
            data_list.append(run_data)
        subj_data = pd.concat(data_list, axis=0)
        subj_data['subject'] = subject
        subj_data['part'] = 2
        subj_list.append(subj_data)
    data = pd.concat(subj_list, axis=0)
    return data


def load_struct(data_dir, subjects=None, onsets=None):
    """Load structure learning data for multiple subjects."""
    if subjects is None:
        subjects = get_subj_list()

    # load raw data for all subjects
    df_all = []
    for subject in subjects:
        df_subj = load_struct_subject(data_dir, subject)
        df_all.append(df_subj)
    raw = pd.concat(df_all, axis=0, ignore_index=True)
    raw['seqtype'].fillna(0, inplace=True)

    # community info
    nodes = network.node_info()
    raw_nodes = nodes.loc[raw['objnum'], :].reset_index()

    # convert to BIDS format
    trial_type = {0: 'structured', 1: 'structured', 2: 'scrambled'}
    orientation = {'cor': 'canonical', 'rot': 'rotated'}
    response = {'c': 'canonical', 'n': 'rotated'}
    object_type = {0: 'central', 1: 'boundary'}
    df = pd.DataFrame(
        {
            'onset': np.nan,
            'duration': 1,
            'trial_type': raw['seqtype'].map(trial_type).astype('category'),
            'subject': raw['SubjNum'],
            'part': raw['part'],
            'run': raw['run'],
            'block': raw['block'],
            'trial': raw['trial'],
            'community': raw_nodes['community'],
            'object': raw['objnum'],
            'object_type': raw_nodes['node_type'].map(object_type).astype('category'),
            'orientation': raw['orientnam'].map(orientation).astype('category'),
            'response': raw['resp'].map(response).astype('category'),
            'response_time': raw['rt'],
            'correct': raw['acc'].astype('Int64'),
        }
    )
    if onsets is not None:
        temp1 = df.set_index(['subject', 'part', 'run', 'trial'])
        temp2 = onsets.set_index(['subject', 'part', 'run', 'trial'])
        temp1['onset'] = temp2['onset']
        temp1['duration'] = temp2['duration']
        df = temp1.reset_index()
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
            'within_opt': 0,
            'response': raw['Resp'],
            'response_time': raw['RT'],
            'correct': raw['Acc'],
        }
    )

    # relabel trials with no response as having NaN accuracy
    df.loc[df['response'].isna(), 'correct'] = np.nan

    # label the "correct" (within-community) response
    opt1_comm = nodes.loc[df['opt1'], 'community'].to_numpy()
    opt2_comm = nodes.loc[df['opt2'], 'community'].to_numpy()
    opt_comm = np.vstack((opt1_comm, opt2_comm))
    i, j = np.nonzero(opt_comm == df['community'].to_numpy())
    within = np.zeros(opt1_comm.shape, dtype=int)
    within[j[i == 0]] = 1
    within[j[i == 1]] = 2
    df['within_opt'] = within

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


def load_parse(data_dir, subjects=None):
    """Load parsing data in BIDs format."""
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

    # set block labels
    n_node = len(nodes)
    n_subject = len(subjects)
    n_run = raw['run'].nunique()
    n_block = int(((len(raw) / n_subject) / n_run) / n_node)
    block = np.tile(
        np.tile(np.repeat(np.arange(1, n_block + 1), n_node), n_run), n_subject
    )

    # convert to BIDS format
    trial_type = {1: 'structured', 2: 'hamiltonian', 3: 'hamiltonian'}
    path_type = {1: 'structured', 2: 'forward', 3: 'backward'}
    response_type = {'PARSED': 1, 'NONE': 0}
    object_type = {0: 'central', 1: 'boundary'}
    df = pd.DataFrame(
        {
            'subject': raw['SubjNum'],
            'run': raw['run'],
            'block': block,
            'trial': raw['trial'],
            'trial_type': raw['objseq'].map(trial_type).astype('category'),
            'path_type': raw['objseq'].map(path_type).astype('category'),
            'community': raw_nodes['community'],
            'object': raw['objnum'],
            'object_type': raw_nodes['node_type'].map(object_type).astype('category'),
            'response': raw['resp'].map(response_type),
            'response_time': raw['rt'],
        }
    )
    return df


def load_group_mat(data_dir, subject_num):
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


def extract_group_coords(mat, subject=None):
    """Extract coordinates from a grouping data matrix."""
    ind = np.where(mat)
    nodes = network.node_info()
    nodes.rename(
        columns={'node': 'object', 'node_type': 'object_type', 'connect': 'connection'},
        inplace=True,
    )
    nodes['object_type'] = nodes['object_type'].map({0: 'central', 1: 'boundary'})
    df_ind = pd.DataFrame(
        {'subject': subject, 'dim1': ind[0], 'dim2': ind[1]}, index=mat[ind]
    )
    df = pd.concat((df_ind, nodes), axis=1)
    return df


def load_group(data_dir, subjects=None):
    """Load grouping task data as a DataFrame."""
    if subjects is None:
        subjects = get_subj_list()

    # load subject data
    df_all = []
    for subject in subjects:
        mat_subj = load_group_mat(data_dir, subject)
        df_subj = extract_group_coords(mat_subj, subject)
        df_all.append(df_subj)
    df = pd.concat(df_all, axis=0, ignore_index=True)
    return df
