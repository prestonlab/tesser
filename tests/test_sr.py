import pytest
import numpy as np
import pandas as pd
from tesser import learn
from tesser import model


@pytest.fixture
def sim1():
    sim = np.array([[1, 2, 3], [3, 2, 1], [1, 3, 2]], dtype='double')
    return sim


@pytest.fixture
def sim2():
    sim = np.array([[3, 2, 1], [1, 3, 2], [2, 1, 3]], dtype='double')
    return sim


@pytest.fixture
def induct_trials():
    trials = {
        'subject': np.array([1, 1, 1, 1, 1, 1]),
        'trial_type': np.array(['1', '1', '1', '2', '2', '2']),
        'cue': np.array([0, 0, 1, 1, 2, 2]),
        'opt1': np.array([1, 1, 0, 0, 0, 0]),
        'opt2': np.array([2, 2, 2, 2, 1, 1]),
        'response': np.array([0, 1, 0, 1, 0, 1]),
    }
    return trials


@pytest.fixture
def induct_cython(induct_trials):
    trials = {
        key: val.astype(dtype=np.dtype('i')) for key, val in induct_trials.items()
    }
    return trials


@pytest.fixture
def induct_pandas(induct_trials):
    trials = induct_trials.copy()
    for key in ['cue', 'opt1', 'opt2', 'response']:
        trials[key] += 1
    induct = pd.DataFrame(trials)
    induct = induct.astype({'trial_type': 'category'})
    return induct


@pytest.fixture
def struct_trials():
    trials = np.array([0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3])
    return trials


@pytest.fixture
def struct_pandas(struct_trials):
    struct = pd.DataFrame(
        {'subject': 1, 'part': 1, 'run': 1, 'object': struct_trials + 1}
    )
    return struct


@pytest.fixture
def struct_sr():
    srm = np.array(
        [
            [0.225, 0.85125, 0.1125, 0.0, 0.0, 0.0],
            [0.25, 0.3375, 0.72625, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.1125, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.225, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.225],
        ]
    )
    return srm


def test_sr_trials(struct_trials, struct_sr):
    """Test SR learning for a set of trials."""
    gamma = 0.9
    alpha = 0.5
    n_state = 6
    SR = np.zeros((n_state, n_state), dtype='double')
    env_step = struct_trials.astype(np.dtype('i'))
    learn.learn_sr(SR, env_step, gamma, alpha)
    np.testing.assert_allclose(SR, struct_sr)


def test_sr_struct(struct_pandas, struct_sr):
    """Test SR learning of structure data."""
    gamma = 0.9
    alpha = 0.5
    n_state = 6
    SR = model.learn_struct_sr(struct_pandas, gamma, alpha, n_state)
    np.testing.assert_allclose(SR, struct_sr)


def test_choice_prob_sim1(sim1):
    """Test choice probability based on one similarity matrix."""
    cue = 0
    opt1 = 1
    opt2 = 2
    response = 0

    tau = 1
    prob = learn.prob_choice_sim(cue, opt1, opt2, response, sim1, tau)
    np.testing.assert_allclose(prob, 0.26894142)

    tau = 0.5
    prob = learn.prob_choice_sim(cue, opt1, opt2, response, sim1, tau)
    np.testing.assert_allclose(prob, 0.11920292)


def test_choice_prob_sim2(sim1, sim2):
    """Test choice probability based on two similarity matrices."""
    cue = 0
    opt1 = 1
    opt2 = 2
    response = 0

    tau = 1
    w = 0.5
    prob = learn.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.5)

    w = 1
    prob = learn.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.26894142)

    w = 0
    prob = learn.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.731058578)


def test_induct_prob_sim(sim1, induct_cython):
    """Test induction test probability based on a similarity matrix."""
    t = induct_cython
    tau = 1
    trial_prob = learn.prob_induct_sim(
        t['cue'], t['opt1'], t['opt2'], t['response'], sim1, tau
    )
    expected = np.array(
        [0.26894142, 0.73105858, 0.88079708, 0.11920292, 0.11920292, 0.88079708]
    )
    np.testing.assert_allclose(trial_prob, expected)


def test_induct_prob_sim2(sim1, sim2, induct_cython):
    """Test induction test probability based on a similarity matrix."""
    t = induct_cython
    tau = 1
    w = 0.5
    trial_prob = learn.prob_induct_sim2(
        t['cue'], t['opt1'], t['opt2'], t['response'], sim1, sim2, w, tau
    )
    expected = np.array([0.5, 0.5, 0.62245933, 0.37754067, 0.37754067, 0.62245933])
    np.testing.assert_allclose(trial_prob, expected)


def test_induct_sim1(sim1, induct_pandas):
    """Test induction test probability from DataFrame."""
    tau = 1
    trial_prob = model.prob_induct(induct_pandas, tau, sim1)
    expected = np.array(
        [0.26894142, 0.73105858, 0.88079708, 0.11920292, 0.11920292, 0.88079708]
    )
    np.testing.assert_allclose(trial_prob, expected)


def test_induct_sim2(sim1, sim2, induct_pandas):
    """Test induction test probability from DataFrame with two matrices."""
    tau = 1
    w = 0.5
    trial_prob = model.prob_induct(induct_pandas, tau, sim1, w, sim2)
    expected = np.array([0.5, 0.5, 0.62245933, 0.37754067, 0.37754067, 0.62245933])
    np.testing.assert_allclose(trial_prob, expected)


def test_prob_struct_induct(struct_pandas, induct_pandas):
    """Test induction test probability given structure learning."""
    param = {'gamma': 0.9, 'alpha': 0.5, 'tau': 1}
    sim_spec = {'alpha': 'alpha', 'gamma': 'gamma'}
    prob = model.prob_struct_induct(struct_pandas, induct_pandas, param, sim_spec)
    expected = np.array(
        [0.67672246, 0.32327754, 0.38313802, 0.61686198, 0.4378235, 0.5621765]
    )
    np.testing.assert_allclose(prob, expected)


def test_prob_struct_induct_sr2(struct_pandas, induct_pandas):
    """Test induction test probability with two SR matrices."""
    param = {
        'alpha1': 0.5, 'gamma1': 0.9, 'alpha2': 0.7, 'gamma2': 0.8, 'tau': 1, 'w': 1
    }
    sim1_spec = {'alpha': 'alpha1', 'gamma': 'gamma1'}
    sim2_spec = {'alpha': 'alpha2', 'gamma': 'gamma2'}
    prob = model.prob_struct_induct(
        struct_pandas, induct_pandas, param, sim1_spec, sim2_spec
    )
    expected = np.array(
        [0.67672246, 0.32327754, 0.38313802, 0.61686198, 0.4378235, 0.5621765]
    )
    np.testing.assert_allclose(prob, expected)


def test_prob_struct_induct_question(struct_pandas, induct_pandas):
    """Test induction test probability with question-specific parameters."""
    param = {
        'alpha1': 0.5,
        'gamma1': 0.9,
        'alpha2': 0.7,
        'gamma2': 0.8,
        'tau': 1,
        'w1': 1,
        'w2': 1,
    }
    sim1_spec = {'alpha': 'alpha1', 'gamma': 'gamma1'}
    sim2_spec = {'alpha': 'alpha2', 'gamma': 'gamma2'}
    question_param = {'1': {'w': 'w1'}, '2': {'w': 'w2'}}
    prob = model.prob_struct_induct(
        struct_pandas,
        induct_pandas,
        param,
        sim1_spec,
        sim2_spec,
        question_param=question_param,
    )
    expected = np.array(
        [0.67672246, 0.32327754, 0.38313802, 0.61686198, 0.4378235, 0.5621765]
    )
    np.testing.assert_allclose(prob, expected)


def test_prob_struct_induct_sim2(struct_pandas, induct_pandas):
    """Test induction test probability with a custom matrix."""
    param = {'alpha1': 0.5, 'gamma1': 0.9, 'tau': 1, 'w': 1}
    sim1_spec = {'alpha': 'alpha1', 'gamma': 'gamma1'}
    sim = np.ones((6, 6))
    sim2_spec = {'sim': sim}
    prob = model.prob_struct_induct(
        struct_pandas, induct_pandas, param, sim1_spec, sim2_spec
    )
    expected = np.array(
        [0.67672246, 0.32327754, 0.38313802, 0.61686198, 0.4378235, 0.5621765]
    )
    np.testing.assert_allclose(prob, expected)


@pytest.fixture
def struct_fit():
    struct = pd.DataFrame(
        {
            'subject': 1,
            'part': 1,
            'run': np.repeat(np.arange(1, 5), 6),
            'object': [1, 3, 5, 7, 2, 4, 6, 8] * 3,
        }
    )
    return struct


@pytest.fixture
def induct_fit():
    mat = [
        [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6] * 2,
        [3, 3, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7] * 2,
        [4, 4, 6, 6, 8, 8, 6, 6, 8, 8, 8, 8] * 2,
        [1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2] * 2,
    ]
    induct = pd.DataFrame(
        {
            'subject': 1,
            'trial_type': '1',
            'cue': mat[0],
            'opt1': mat[1],
            'opt2': mat[2],
            'response': mat[3],
        }
    )
    induct = induct.astype({'trial_type': 'category'})
    return induct


def test_fit_induct(struct_fit, induct_fit):
    """Test fitting model to induction data."""
    fixed = {'tau': 1}
    var_names = ['alpha', 'gamma']
    var_bounds = {'alpha': [0, 1], 'gamma': [0, 1]}
    sim1_spec = {'alpha': 'alpha', 'gamma': 'gamma'}
    logl, param = model._fit_induct(
        struct_fit, induct_fit, fixed, var_names, var_bounds, sim1_spec
    )
    np.testing.assert_allclose(param['alpha'], 0.857, atol=0.01)
    np.testing.assert_allclose(param['gamma'], 0.769, atol=0.01)


def test_fit_induct_indiv(struct_fit, induct_fit):
    """Test fitting model to induction subject data."""
    fixed = {'tau': 1}
    var_names = ['alpha', 'gamma']
    var_bounds = {'alpha': [0, 1], 'gamma': [0, 1]}
    sim1_spec = {'alpha': 'alpha', 'gamma': 'gamma'}
    results = model.fit_induct_indiv(
        struct_fit, induct_fit, fixed, var_names, var_bounds, sim1_spec
    )
    np.testing.assert_allclose(results.loc[1, 'alpha'], 0.857, atol=0.01)
    np.testing.assert_allclose(results.loc[1, 'gamma'], 0.769, atol=0.01)
