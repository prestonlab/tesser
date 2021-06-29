"""Utilities for loading Tesser behavioral data."""

import numpy as np
from scipy.spatial import distance
from scipy import stats
from sklearn import linear_model
import pandas as pd
import os
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


def _load_bids_events_subject(bids_dir, phase, data_type, task, subject, runs=None):
    """Load events for one subject from a BIDS directory."""
    subj_dir = os.path.join(bids_dir, f'sub-{subject}', phase)
    if not os.path.exists(subj_dir):
        raise IOError(f'Subject directory does not exist: {subj_dir}')

    if runs is not None:
        df_list = []
        for run in runs:
            file = os.path.join(
                subj_dir, f'sub-{subject}_task-{task}_run-{run}_{data_type}.tsv'
            )
            df_run = pd.read_table(file)
            df_run['run'] = run
            df_list.append(df_run)
        data = pd.concat(df_list, axis=0)
    else:
        file = os.path.join(
            subj_dir, f'sub-{subject}_task-{task}_{data_type}.tsv'
        )
        data = pd.read_table(file)
        data['run'] = 1
    data['subject'] = subject
    return data


def _load_bids_events(bids_dir, phase, data_type, task, subjects=None, runs=None):
    """Load events for multiple subjects from a BIDS directory."""
    if subjects is None:
        subjects = get_subj_list()
    elif not isinstance(subjects, list):
        subjects = [subjects]

    data = pd.concat(
        [
            _load_bids_events_subject(
                bids_dir, phase, data_type, task, subject, runs
            ) for subject in subjects
        ], axis=0
    )
    return data


def load_struct(bids_dir, subjects=None):
    """Load structure learning events."""
    if subjects is None:
        subjects = get_subj_list()
    elif not isinstance(subjects, list):
        subjects = [subjects]

    data_list = []
    for subject in subjects:
        learn = _load_bids_events_subject(
            bids_dir, 'beh', 'events', 'learn', subject, runs=list(range(1, 6))
        )
        struct = _load_bids_events_subject(
            bids_dir, 'func', 'events', 'struct', subject, runs=list(range(1, 7))
        )
        subj_data = pd.concat([learn, struct], axis=0)
        data_list.append(subj_data)
    data = pd.concat(data_list, axis=0)
    return data


def load_induct(bids_dir, subjects=None):
    """Load induction test events."""
    data = _load_bids_events(bids_dir, 'beh', 'events', 'induct', subjects)
    return data


def load_parse(bids_dir, subjects=None):
    """Load parsing task events."""
    data = _load_bids_events(
        bids_dir, 'beh', 'events', 'parse', subjects, runs=list(range(1, 4))
    )
    return data


def load_group(bids_dir, subjects=None):
    """Load grouping task events."""
    data = _load_bids_events(bids_dir, 'beh', 'beh', 'group', subjects)
    return data


def fix_struct_switched(raw):
    """Fix responses that appear to have been switched."""
    # on these two scanning runs, d-prime is substantially negative,
    # suggesting buttons were confused
    switched = [[120, 2, 1], [122, 2, 3]]
    data = raw.copy()
    for subject, part, run in switched:
        include = raw.eval(
            f'subject == {subject} and part == {part} and run == {run}')
        response = raw.loc[include, 'response']
        data.loc[include & (response == 'canonical'), 'response'] = 'rotated'
        data.loc[include & (response == 'rotated'), 'response'] = 'canonical'
    return data


def response_zscore(n, m):
    """Z-score of response rate."""
    rate = n / m
    if isinstance(n, np.ndarray):
        rate[rate == 0] = 0.5 / m
        rate[rate == 1] = (m - 0.5) / m
    else:
        if rate == 0:
            rate = 0.5 / m
        elif rate == 1:
            rate = (m - 0.5) / m
    z = stats.norm.ppf(rate)
    return z


def rotation_perf(data):
    """Calculate performance for rotation task data."""
    response = data['response'].notna()
    n_can = np.count_nonzero(response & (data['orientation'] == 'canonical'))
    n_rot = np.count_nonzero(response & (data['orientation'] == 'rotated'))
    n_hit = np.count_nonzero(
        (data['orientation'] == 'rotated') & (data['response'] == 'rotated')
    )
    n_fa = np.count_nonzero(
        (data['orientation'] == 'canonical') & (data['response'] == 'rotated')
    )
    rr = np.count_nonzero(response) / len(data)
    hr = n_hit / n_rot
    far = n_fa / n_can
    zhr = response_zscore(n_hit, n_rot)
    zfar = response_zscore(n_fa, n_can)
    dprime = zhr - zfar
    res = pd.Series(
        {'rr': rr, 'hr': hr, 'far': far, 'zhr': zhr, 'zfar': zfar, 'dprime': dprime}
    )
    return res


def test_rotation_perf(data, n_perm):
    """Test whether rotation task peformance is above chance."""
    # trial types and responses
    made_response = data['response'].notna()
    response = data.loc[made_response, 'response'].to_numpy()
    orientation = data.loc[made_response, 'orientation'].to_numpy()

    # scramble trial responses
    n_item = len(response)
    rand_ind = [
        np.random.choice(np.arange(n_item), n_item, False) for i in range(n_perm - 1)
    ]
    rand_ind.insert(0, np.arange(n_item))
    rand_ind = np.array(rand_ind)
    response_perm = response[rand_ind]

    # hit rate and false alarm rate
    n_can = np.count_nonzero(orientation == 'canonical')
    n_rot = np.count_nonzero(orientation == 'rotated')
    n_hit = np.count_nonzero(
        (orientation == 'rotated') & (response_perm == 'rotated'), axis=1
    )
    n_fa = np.count_nonzero(
        (orientation == 'canonical') & (response_perm == 'rotated'), axis=1
    )
    rr = np.count_nonzero(response) / len(data)
    hr = n_hit / n_rot
    far = n_fa / n_can

    # calculate dprime
    zhr = response_zscore(n_hit, n_rot)
    zfar = response_zscore(n_fa, n_can)
    dprime = zhr - zfar

    # significance based on permutation test
    p = np.mean(dprime >= dprime[0])
    res = pd.Series(
        {
            'rr': rr,
            'hr': hr[0],
            'far': far[0],
            'zhr': zhr[0],
            'zfar': zfar[0],
            'dprime': dprime[0],
            'p': p,
        }
    )
    return res


def score_parse(parse):
    """Score parsing task data."""
    # number walks within a community
    parse = parse.copy()
    parse['transition'] = parse.groupby(['subject', 'run'])['community'].transform(
        lambda data: data.diff().fillna(0).abs().astype(bool)
    )
    parse['walk'] = parse.groupby(['subject', 'run'])['transition'].transform(
        lambda data: data.astype('Int64').cumsum() + 1
    )
    walk_lengths = parse.groupby(['subject', 'run', 'walk'])['walk'].count()

    # length of the previous walk
    parse['prev_walk'] = np.nan
    for (subject, run, walk), walk_length in walk_lengths.iteritems():
        include = (
            (parse['subject'] == subject) &
            (parse['run'] == run) &
            (parse['walk'] == (walk + 1))
        )
        if any(include):
            parse.loc[include.to_numpy(), 'prev_walk'] = walk_length
    return parse


def parse_perf(parse):
    """Score parsing performance."""
    trans_parse = (
        parse.query('transition and prev_walk >= 4')
             .groupby(['subject', 'trial_type'])['response']
             .mean()
    )
    other_parse = (
        parse.query('~(transition and prev_walk >= 4)')
             .groupby(['subject', 'trial_type'])['response']
             .mean()
    )
    results = pd.concat([trans_parse, other_parse], keys=['transition', 'other'])
    results.index.set_names('parse_type', level=0, inplace=True)
    results = results.reset_index()
    parse_type = results['parse_type'].astype('category')
    parse_type = parse_type.cat.set_categories(['transition', 'other'])
    results['parse_type'] = parse_type
    return results


def load_bids_parse(bids_dir, subject):
    """Load parsing task data from BIDS files."""
    runs = np.arange(1, 4)
    df_list = []
    for run in runs:
        file = os.path.join(
            bids_dir,
            f'sub-{subject}',
            'beh',
            f'sub-{subject}_task-parse_run-{run}_events.tsv',
        )
        df_run = pd.read_table(file)
        df_run['run'] = run
        df_list.append(df_run)
    data = pd.concat(df_list, axis=0)
    return data


def group_rdms(data):
    """Calculate dissimilarity matrices from grouping task data."""
    rdms = {}
    for subject, df in data.groupby('subject'):
        coords = df.filter(like='dim').to_numpy()
        rdm = distance.squareform(distance.pdist(coords, 'euclidean'))
        rdms[subject] = rdm
    return rdms


def group_distance(data):
    """Calculate within and across group distance."""
    # calculate pairwise distances (stored as vectors)
    rdv_list = []
    for _, df in data.groupby('subject'):
        coords = df.filter(like='dim').to_numpy()
        subj_rdv = distance.pdist(coords, 'euclidean')
        rdv_list.append(subj_rdv)
    rdv = np.vstack(rdv_list)

    # get a vector that is ones for within-community pairs
    nodes = network.node_info()
    comm = nodes['community'].to_numpy()
    within = comm == comm[:, None]
    within_vec = distance.squareform(within, checks=False)

    # calculate distance stats
    subject = data['subject'].unique()
    m_within = np.mean(rdv[:, within_vec == 1], axis=1)
    m_across = np.mean(rdv[:, within_vec == 0], axis=1)
    res = pd.DataFrame({'subject': subject, 'within': m_within, 'across': m_across})
    return res


def resid(x, y):
    """Residual of y after regressing out x."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    model = linear_model.LinearRegression()
    model.fit(x, y)
    yhat = model.predict(x)
    res = y - yhat
    return res
