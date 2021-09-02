"""Representational similarity analysis of objects after learning."""

import os
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
import nibabel as nib
import sklearn.linear_model as lm
from nilearn.glm import first_level


def load_cluster_patterns(beta_dir, subject, roi, contrast, stat, cluster, dilate):
    """Load cluster community similarity stats."""
    # load pattern
    mat_file = os.path.join(
        beta_dir,
        roi,
        contrast,
        'clusters',
        f'sub-{subject}_desc-{stat}{cluster}dil{dilate}_beta.npy',
    )
    pattern = np.load(mat_file)

    # load events
    events_file = os.path.join(
        beta_dir, roi, contrast, 'clusters', f'sub-{subject}_events.tsv'
    )
    events = pd.read_table(events_file)

    # subtract run pattern
    z_pattern = np.zeros(pattern.shape)
    run = events['run'].to_numpy()
    runs = events['run'].unique()
    for r in runs:
        samples = run == r
        z_pattern[samples] = pattern[samples] - np.mean(pattern[samples], 0)
    return z_pattern, events


def beta_sim_stats(events, patterns):
    """Calculate similarity statistics from betaseries patterns."""
    # create matrices labeling event pairs
    run = events['run'].to_numpy()
    item = events['object'].to_numpy()
    community = events['community'].to_numpy()
    across_run = run != run[:, np.newaxis]
    within_community = community == community[:, np.newaxis]
    across_community = community != community[:, np.newaxis]
    diff_object = item != item[:, np.newaxis]
    lower = np.ones(across_run.shape, dtype=bool)
    lower = np.tril(lower, -1)

    # calculate an RDM for each pattern
    rdms = [sd.squareform(sd.pdist(pattern, 'correlation')) for pattern in patterns]

    # calculate average similarity for each bin
    include = across_run & within_community & diff_object & lower
    sim_within = np.array([np.mean(1 - rdm[include]) for rdm in rdms])
    include = across_run & across_community & diff_object & lower
    sim_across = np.array([np.mean(1 - rdm[include]) for rdm in rdms])
    sim = {'within': sim_within, 'across': sim_across}
    return sim


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


def prepare_betaseries_design(
    events_file, conf_file, tr, high_pass, exclude_motion=False
):
    """Prepare betaseries design matrix and confounds."""
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

    # exclude motion outliers
    if exclude_motion:
        outliers = conf.filter(like='motion_outlier')
        if not outliers.empty:
            nuisance = np.hstack([nuisance, outliers.to_numpy()])

    # create design matrix
    n_sample = len(conf)
    events = pd.read_csv(events_file, sep='\t')
    design = create_betaseries_design(events, n_sample, tr, high_pass)
    n_object = events['object'].nunique()
    mat = design.iloc[:, :n_object].to_numpy()
    confound = np.hstack((design.iloc[:, n_object:-1].to_numpy(), nuisance))
    return mat, confound


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
        dm_full = dm_full - np.kron(np.ones(s), np.mean(dm_full, 0))[: s[0], : s[1]]
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
    base = f'sub-{subject}_task-{task}_run-{run}_space-{space}'
    if desc is not None:
        mask_name = f'{base}_desc-{desc}_mask.nii.gz'
    else:
        mask_name = f'{base}_label-{label}_probseg.nii.gz'
    mask_file = os.path.join(base_dir, f'sub-{subject}', 'func', mask_name)
    return mask_file


def get_anat_mask(base_dir, subject, space, desc=None, label=None):
    """Get the full path to an anatomical mask."""
    base = f'sub-{subject}_space-{space}'
    if desc is not None:
        mask_name = f'{base}_desc-{desc}_mask.nii.gz'
    else:
        mask_name = f'{base}_label-{label}_probseg.nii.gz'
    mask_file = os.path.join(base_dir, f'sub-{subject}', 'anat', mask_name)
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
    exclude_motion=False,
):
    """Estimate betaseries for one run."""
    tr = 2
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
    mat, confound = prepare_betaseries_design(
        events_file, conf_file, tr, high_pass, exclude_motion
    )

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

    # estimate model residuals (for smoothness calculation)
    model = lm.LinearRegression()
    design = np.hstack([mat, confound])
    model.fit(design, data)
    resid = data - model.predict(design)
    return beta, resid
