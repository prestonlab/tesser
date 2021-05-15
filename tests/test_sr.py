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
        'cue': np.array([0, 0, 1, 1, 2, 2]),
        'opt1': np.array([1, 1, 0, 0, 0, 0]),
        'opt2': np.array([2, 2, 2, 2, 1, 1]),
        'response': np.array([0, 1, 0, 1, 0, 1]),
    }
    return trials


@pytest.fixture
def induct_cython(induct_trials):
    trials = {}
    for key, val in induct_trials.items():
        trials[key] = induct_trials[key].astype(dtype=np.dtype('i'))
    return trials


@pytest.fixture
def induct_pandas(induct_trials):
    trials = pd.DataFrame(induct_trials)
    return trials


@pytest.fixture
def struct_trials():
    trials = np.array([0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3])
    return trials


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
