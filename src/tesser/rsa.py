"""Representational similarity analysis of objects after learning."""

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from scipy import linalg
from scipy import stats
import nibabel as nib
from nilearn.glm import first_level
from mindstorm import prsa
from tesser import tasks
from tesser import network
from tesser import learn


def get_roi_sets():
    """Get a list of rois included in a set."""
    rois = {
        'hpc3': ['b_hip_ant', 'b_hip_body', 'b_hip_tail'],
        'hpc3b': [
            'r_hip_ant',
            'r_hip_body',
            'r_hip_tail',
            'l_hip_ant',
            'l_hip_body',
            'l_hip_tail',
        ],
        'mpfc9': ['10m', '10p', '10r', '11m', '14c', '14r', '24', '25', '32pl'],
        'mpfc3': ['ampfc', 'mmpfc', 'pmpfc'],
        'ifg3': ['b_oper', 'b_tria', 'b_orbi'],
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
            study_dir,
            'batch',
            'analysis',
            'rsa_beta',
            'rsa_event_textfiles',
            f'tesser_{subject}_run{run}_info.txt',
        )
        run_data = pd.read_csv(vol_file, names=columns)
        run_data['duration'] = 1
        run_data['run'] = run
        data_list.append(run_data)
    data = pd.concat(data_list, axis=0)
    return data


def load_struct_vols(bids_dir, subject):
    """Load beta volume information for a subject."""
    vols_list = []
    columns = ['duration', 'run', 'trial_type', 'community', 'object', 'object_type']
    for run in range(1, 7):
        events_file = os.path.join(
            bids_dir,
            f'sub-{subject}',
            'func',
            f'sub-{subject}_task-struct_run-{run}_events.tsv',
        )
        events = pd.read_table(events_file)
        run_vols = (
            events.query('trial_type == "scrambled"')
            .groupby('object')
            .first()
            .reset_index()
        )
        run_vols['run'] = run
        vols_list.append(run_vols.loc[:, columns])
    vols = pd.concat(vols_list, axis=0)
    return vols


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
        subjects = tasks.get_subj_list()

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
            np.load(os.path.join(res_dir, roi, f'sub-{subject}_brsa.npz'))['C'][
                ind, ind
            ]
            for subject in subjects
        ]
        for roi in rois
    }
    return rdms


def load_roi_mean_brsa(res_dir, rois, blocks=None, subjects=None):
    """Load mean correlation matrices from BRSA results."""
    if subjects is None:
        subjects = tasks.get_subj_list()
    rdms = load_roi_brsa(res_dir, rois, blocks, subjects)
    n_subj = len(subjects)
    mrdm = {
        roi: np.mean(
            np.dstack([rdms[roi][subject] for subject in range(n_subj)]), axis=2
        )
        for roi in rois
    }
    return mrdm


def load_roi_brsa_vec(res_dir, roi, blocks=None, subjects=None):
    """Load correlation matrices from BRSA results."""
    if subjects is None:
        subjects = tasks.get_subj_list()

    rdv_list = []
    for subject in subjects:
        rdm = np.load(os.path.join(res_dir, roi, f'sub-{subject}_brsa.npz'))['C']
        if blocks is not None:
            if blocks == 'walk':
                rdm = rdm[:21, :21]
            elif blocks in ['rand', 'random']:
                rdm = rdm[21:, 21:]
            else:
                raise ValueError(f'Invalid blocks setting: {blocks}')
        rdv = sd.squareform(rdm, checks=False)
        rdv_list.append(rdv)
    rdvs = np.vstack(rdv_list)
    return rdvs


def mean_corr_community(rdvs, subjects):
    """Calculate mean correlations for within and across community."""
    nodes = network.node_info()
    comm = nodes['community'].to_numpy()
    within_mat = comm == comm[:, None]
    within_vec = sd.squareform(within_mat, checks=False)

    node_type = nodes['node_type'].to_numpy()
    central_mat = (node_type == 0) & (node_type[:, None] == 0)
    central_vec = sd.squareform(central_mat, checks=False)

    df_list = []
    for roi, vectors in rdvs.items():
        m_within = np.mean(vectors[:, within_vec == 1], 1)
        m_across = np.mean(vectors[:, within_vec == 0], 1)
        m_within_central = np.mean(vectors[:, (within_vec == 1) & (central_vec == 1)], 1)
        m_across_central = np.mean(vectors[:, (within_vec == 0) & (central_vec == 1)], 1)
        df = pd.DataFrame(
            {
                'within': m_within,
                'across': m_across,
                'diff': m_within - m_across,
                'within_central': m_within_central,
                'across_central': m_across_central,
                'diff_central': m_within_central - m_across_central,
            }, index=subjects
        )
        df_list.append(df)
    results = pd.concat(df_list, keys=rdvs.keys())
    results.index.rename(['roi', 'subject'], inplace=True)
    return results


def beta_sr_rdm(struct, alpha, gamma, distance='correlation'):
    """Calculate an RDM for an SR model of each run."""
    if struct['subject'].nunique() > 1:
        raise ValueError('Structure data should only contain one subject.')

    n_state = struct['object'].nunique()
    M = np.zeros([n_state, n_state])
    M_run = {}
    for ind, df in struct.groupby(['part', 'run']):
        envstep = df['object'].to_numpy() - 1
        envstep = envstep.astype(np.dtype('i'))
        learn.learn_sr(M, envstep, gamma, alpha)
        M_run[ind] = M.copy()
    struct_mat = [M_run[(2, n)] for n in range(1, 7)]
    big_mat = np.vstack(struct_mat)
    big_rdm = sd.squareform(sd.pdist(big_mat, 'correlation'))
    return big_rdm


def beta_community_rdm(struct):
    """Calculate a betaseries RDM for the community model."""
    vols = struct.groupby(['run', 'object']).first().reset_index()
    nodes = network.node_info()
    vol_nodes = nodes.loc[vols['object']]
    community = vol_nodes['community'].to_numpy()
    rdm = community == community[:, np.newaxis]
    return rdm


def load_roi_prsa(res_dir, roi, subjects=None, stat='zstat'):
    """Load z-statistic from permutation test results."""
    if subjects is None:
        subjects = tasks.get_subj_list()

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
        run_events = events.query(f'run == {run}')[['trial_type', 'onset', 'duration']]
        df_run = first_level.make_first_level_design_matrix(
            frame_times, events=run_events, high_pass=high_pass
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


def create_betaseries_design(trials, n_vol, tr, high_pass=0):
    """Create a design matrix for betaseries estimation."""
    # set new EVs with each object in the scrambled run and one for
    # the structured runs
    sequence = trials['trial_type'].to_numpy()
    trial_type = trials['object'].to_numpy().copy()
    n_evs = trials.query('trial_type == "scrambled"')['object'].nunique()

    structured = sequence == 'structured'
    trial_type[structured] = trial_type[structured] + n_evs
    events = trials.filter(['trial_type', 'onset', 'duration'])
    events['trial_type'] = trial_type

    # create a design matrix
    frame_times = np.arange(n_vol) * tr
    design = first_level.make_first_level_design_matrix(
        frame_times, events=events, high_pass=high_pass
    )
    return design


def estimate_betaseries(data, design, confound=None):
    """
    Estimate beta images for a set of trials.

    Parameters
    ----------
    data : numpy.ndarray
        [timepoints x voxels] array of functional data to model.

    design : numpy.ndarray
        [timepoints x EVs] array. Each explanatory variable (EV) will
        be estimated using a separate model.

    confound : numpy.ndarray
        [timepoints x regressors] array with regressors of no interest,
        which will be included in each model.

    Returns
    -------
    beta : numpy.ndarray
        [EVs by voxels] array of beta estimates.
    """
    n_trial = design.shape[1]
    n_sample = data.shape[0]
    beta_maker = np.zeros((n_trial, n_sample))
    trial_evs = list(range(n_trial))
    for i, ev in enumerate(trial_evs):
        # this trial
        dm_trial = design[:, ev, np.newaxis]

        # other trials, summed together
        other_trial_evs = [x for x in trial_evs if x != ev]
        dm_otherevs = np.sum(design[:, other_trial_evs, np.newaxis], 1)

        # put together the design matrix
        if confound is not None:
            dm_full = np.hstack((dm_trial, dm_otherevs, confound))
        else:
            dm_full = np.hstack((dm_trial, dm_otherevs))
        s = dm_full.shape
        dm_full = dm_full - np.kron(np.ones(s), np.mean(dm_full, 0))[:s[0], :s[1]]
        dm_full = np.hstack((dm_full, np.ones((n_sample, 1))))

        # calculate beta-forming vector
        beta_maker_loop = np.linalg.pinv(dm_full)
        beta_maker[i, :] = beta_maker_loop[0, :]

    # this uses Jeanette Mumford's trick of extracting the beta-forming
    # vector for each trial and putting them together, which allows
    # estimation for all trials at once
    beta = np.dot(beta_maker, data)
    return beta


def get_func_mask(base_dir, subject, task, run, space, desc=None, label=None):
    """Get the full path to a functional mask."""
    mask_dir = os.path.join(base_dir, f'sub-{subject}')
    base = f'sub-{subject}_task-{task}_run-{run}_space-{space}'
    if desc is not None:
        mask_file = f'{base}_desc-{desc}_mask.nii.gz'
    else:
        mask_file = f'{base}_label-{label}_probseg.nii.gz'
    return mask_file


def get_anat_mask(base_dir, subject, space, desc=None, label=None):
    """Get the full path to an anatomical mask."""
    mask_dir = os.path.join(base_dir, f'sub-{subject}')
    base = f'sub-{subject}_space-{space}'
    if desc is not None:
        mask_file = f'{base}_desc-{desc}_mask.nii.gz'
    else:
        mask_file = f'{base}_label-{label}_probseg.nii.gz'
    return mask_file


def run_betaseries(
    raw_dir,
    post_dir,
    mask,
    bold,
    subject,
    run,
    high_pass=0,
    space='T1w',
    mask_dir='func',
    mask_thresh=None,
):
    """Estimate betaseries for one run."""
    tr = 2
    n_object = 21
    subj_raw = os.path.join(raw_dir, f'sub-{subject}', 'func')
    subj_post = os.path.join(post_dir, f'sub-{subject}', 'func')

    # task events
    events_file = os.path.join(
        subj_raw, f'sub-{subject}_task-struct_run-{run}_events.tsv'
    )
    if not os.path.exists(events_file):
        raise IOError(f'Events do not exist: {events_file}')

    # ROI/brain mask
    if mask_dir == 'func':
        mask_file = get_func_mask(post_dir, subject, 'struct', run, space, desc=mask)
    else:
        mask_file = get_anat_mask(post_dir, subject, space, label=mask)
    if not os.path.exists(mask_file):
        raise IOError(f'Mask file does not exist: {mask_file}')

    # BOLD scan
    bold_file = os.path.join(
        subj_post,
        f'sub-{subject}_task-struct_run-{run}_space-{space}_desc-{bold}_bold.nii.gz',
    )
    if not os.path.exists(bold_file):
        raise IOError(f'BOLD file does not exist: {bold_file}')

    # confounds file
    conf_file = os.path.join(
        subj_post, f'sub-{subject}_task-struct_run-{run}_desc-confounds_timeseries.tsv'
    )
    if not os.path.exists(conf_file):
        raise IOError(f'Confounds file does not exist: {conf_file}')

    # create nuisance regressor matrix
    conf = pd.read_csv(conf_file, sep='\t')
    include = [
        'csf',
        'csf_derivative1',
        'white_matter',
        'white_matter_derivative1',
        'trans_x',
        'trans_x_derivative1',
        'trans_y',
        'trans_y_derivative1',
        'trans_z',
        'trans_z_derivative1',
        'rot_x',
        'rot_x_derivative1',
        'rot_y',
        'rot_y_derivative1',
        'rot_z',
        'rot_z_derivative1',
    ]
    raw = conf.filter(include).to_numpy()
    nuisance = raw - np.nanmean(raw, 0)
    nuisance[np.isnan(nuisance)] = 0

    # create design matrix
    n_sample = len(conf)
    events = pd.read_csv(events_file, sep='\t')
    design = create_betaseries_design(events, n_sample, tr, high_pass)
    mat = design.iloc[:, :n_object].to_numpy()
    confound = np.hstack((design.iloc[:, n_object:-1].to_numpy(), nuisance))

    # load functional data
    bold_vol = nib.load(bold_file)
    mask_vol = nib.load(mask_file)
    bold_img = bold_vol.get_fdata()
    if mask_thresh is None:
        mask_img = mask_vol.get_fdata().astype(bool)
    else:
        mask_img = mask_vol.get_fdata() > mask_thresh
    data = bold_img[mask_img].T

    # estimate each beta image
    beta = estimate_betaseries(data, mat, confound)
    return beta
