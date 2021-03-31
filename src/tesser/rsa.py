"""Representational similarity analysis of objects after learning."""

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from scipy import linalg
from scipy import stats
from nilearn.glm import first_level
from mindstorm import prsa
from tesser import util


def get_roi_sets():
    """Get a list of rois included in a set."""
    rois = {
        'hpc3': ['b_hip_ant', 'b_hip_body', 'b_hip_tail'],
        'mpfc9': ['10m', '10p', '10r', '11m', '14c', '14r', '24', '25', '32pl'],
        'mpfc3': ['ampfc', 'mmpfc', 'pmpfc'],
    }
    return rois


def parse_rois(roi_list):
    """Parse an roi spec list."""
    roi_sets = get_roi_sets()
    split_rois = roi_list.split(',')
    rois = []
    for roi in split_rois:
        if roi in roi_sets:
            rois.extend(roi_sets[roi])
        else:
            rois.append(roi)
    return rois


def load_vol_info(study_dir, subject):
    """Load volume information for all runs for a subject."""
    data_list = []
    columns = ['trial', 'onset', 'tr', 'sequence_type', 'trial_type', 'duration']
    runs = list(range(1, 7))
    for i, run in enumerate(runs):
        vol_file = os.path.join(
            study_dir, 'batch', 'analysis', 'rsa_beta', 'rsa_event_textfiles',
            f'tesser_{subject}_run{run}_info.txt'
        )
        run_data = pd.read_csv(vol_file, names=columns)
        run_data['duration'] = 1
        run_data['run'] = run
        data_list.append(run_data)
    data = pd.concat(data_list, axis=0)
    return data


def make_sym_matrix(asym_mat):
    """Calculate an average symmetric matrix from an asymmetric matrix."""
    v1 = sd.squareform(asym_mat, checks=False)
    v2 = sd.squareform(asym_mat.T, checks=False)
    vm = (v1 + v2) / 2
    sym_mat = sd.squareform(vm)
    return sym_mat


def load_roi_brsa(res_dir, rois, blocks=None, subjects=None):
    """Load correlation matrices from BRSA results."""
    if subjects is None:
        subjects = util.get_subj_list()

    if blocks is None:
        ind = slice(None, None)
    elif blocks == 'walk':
        ind = slice(None, 21)
    elif blocks in ['rand', 'random']:
        ind = slice(21, None)
    else:
        raise ValueError(f'Invalid blocks setting: {blocks}')

    rdms = {
        roi: [
            np.load(
                os.path.join(res_dir, roi, f'sub-{subject}_brsa.npz')
            )['C'][ind, ind]
            for subject in subjects
        ] for roi in rois
    }
    return rdms


def load_roi_mean_brsa(res_dir, rois, blocks=None, subjects=None):
    """Load mean correlation matrices from BRSA results."""
    if subjects is None:
        subjects = util.get_subj_list()
    rdms = load_roi_brsa(res_dir, rois, blocks, subjects)
    n_subj = len(subjects)
    mrdm = {
        roi: np.mean(
            np.dstack([rdms[roi][subject] for subject in range(n_subj)]), axis=2
        ) for roi in rois
    }
    return mrdm


def load_roi_brsa_vec(res_dir, roi, blocks=None, subjects=None):
    """Load correlation matrices from BRSA results."""
    if subjects is None:
        subjects = util.get_subj_list()

    rdv_list = []
    for subject in subjects:
        rdm = np.load(os.path.join(res_dir, roi, f'sub-{subject}_brsa.npz'))['C']
        if blocks is not None:
            if blocks == 'walk':
                rdm = rdm[:21, :21]
            elif blocks == 'random':
                rdm = rdm[21:, 21:]
            else:
                raise ValueError(f'Invalid blocks setting: {blocks}')
        rdv = sd.squareform(rdm, checks=False)
        rdv_list.append(rdv)
    rdvs = np.vstack(rdv_list)
    return rdvs


def load_roi_prsa(res_dir, roi, subjects=None, stat='zstat'):
    """Load z-statistic from permutation test results."""
    if subjects is None:
        subjects = util.get_subj_list()

    z = []
    for subject in subjects:
        subj_file = os.path.join(res_dir, roi, f'zstat_{subject}.csv')
        zdf = pd.read_csv(subj_file, index_col=0).T
        sdf = zdf.loc[[stat]]
        sdf.index = [subject]
        z.append(sdf)
    df = pd.concat(z)
    return df


def load_net_prsa(
    rsa_dir, brsa_name, block, model_set, rois, subjects=None, stat='zstat'
):
    """Load z-statistics for a model for a set of ROIs."""
    # get the directory with results for this model set and block
    res_dir = os.path.join(rsa_dir, f'{brsa_name}_{block}_{model_set}')
    if not os.path.exists(res_dir):
        raise IOError(f'Results directory not found: {res_dir}')

    # load each ROI
    df_list = []
    for roi in rois:
        rdf = load_roi_prsa(res_dir, roi, subjects, stat=stat)
        mdf = pd.DataFrame({'subject': rdf.index, 'roi': roi}, index=rdf.index)
        full = pd.concat((mdf, rdf), axis=1)

        df_list.append(full)
    df = pd.concat(df_list, ignore_index=True)
    return df


def net_prsa_perm(df, model, n_perm=1000, beta=0.05):
    """Test ROI correlations using a permutation test."""
    # shape into matrix format
    rois = df.roi.unique()
    mat = df.pivot(index='subject', columns='roi', values=model)
    mat = mat.reindex(columns=rois)

    # run sign flipping test
    results = prsa.sign_perm(mat.to_numpy(), n_perm, beta)
    results.index = mat.columns
    return results


def create_brsa_matrix(
    subject_dir, events, n_vol, high_pass=0, censor=False, baseline=False
):
    """Create a design matrix for Bayesian RSA."""
    # load confound files
    runs = events['run'].unique()
    n_run = len(runs)
    confound = {}
    for run in runs:
        confound_file = os.path.join(
            subject_dir, 'BOLD', f'functional_run_{run}', 'QA', 'confound.txt'
        )
        confound[run] = np.loadtxt(confound_file)

    # explanatory variables of interest
    n_ev = events['trial_type'].nunique()
    evs = np.arange(1, n_ev + 1)

    # create full design matrix
    signal_list = []
    confound_list = []
    n_run_vol = n_vol // n_run
    frame_times = np.arange(n_run_vol) * 2
    scan_onsets = np.arange(0, n_vol, n_run_vol, dtype=int)
    for run in runs:
        # create a design matrix with one column per trial type and confounds
        df_run = first_level.make_first_level_design_matrix(
            frame_times, events=events.query(f'run == {run}'), high_pass=high_pass
        )

        # signals by themselves (set any missing evs to zero)
        signal_df = df_run.reindex(columns=evs)
        signal_df.fillna(0, inplace=True)
        signal_list.append(signal_df.to_numpy())

        # assuming 6 motion + 6 derivatives + FD and DVARS
        n_nuisance = 14

        # nuisance regressors for this run
        run_confounds = [confound[run][:, :n_nuisance]]
        if high_pass > 0:
            drifts = df_run.filter(like='drift', axis=1).to_numpy()
            run_confounds.append(drifts)

        if censor:
            run_confounds.append(confound[run][:, n_nuisance:])

        confound_mat = stats.zscore(np.hstack(run_confounds), 0)
        if baseline:
            n_vol = confound[run].shape[0]
            confound_mat = np.hstack((np.ones((n_vol, 1)), confound_mat))
        confound_list.append(confound_mat)

    # make full confound matrix
    nuisance = linalg.block_diag(*confound_list)

    # package for use with BRSA
    mat = np.vstack(signal_list)
    return mat, nuisance, scan_onsets
