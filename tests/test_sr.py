import pytest
import numpy as np
from tesser import sr


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
        'cue': np.array([0, 0, 1, 1, 2, 2], dtype=np.dtype('i')),
        'opt1': np.array([1, 1, 0, 0, 0, 0], dtype=np.dtype('i')),
        'opt2': np.array([2, 2, 2, 2, 1, 1], dtype=np.dtype('i')),
        'response': np.array([0, 1, 0, 1, 0, 1], dtype=np.dtype('i')),
    }
    return trials


def test_sr_trials():
    """Test SR learning for a set of trials."""
    gamma = 0.9
    alpha = 0.5
    n_state = 6
    SR = np.zeros((n_state, n_state), dtype='double')
    env_step = np.array([0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3], dtype=np.dtype('i'))
    sr.learn_sr(SR, env_step, gamma, alpha)

    expected = np.array(
        [
            [0.225, 0.85125, 0.1125, 0.0, 0.0, 0.0],
            [0.25, 0.3375, 0.72625, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.1125, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.225, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.225],
        ]
    )
    np.testing.assert_allclose(SR, expected)


def test_choice_prob_sim1(sim1):
    """Test choice probability based on one similarity matrix."""
    cue = 0
    opt1 = 1
    opt2 = 2
    response = 0

    tau = 1
    prob = sr.prob_choice_sim(cue, opt1, opt2, response, sim1, tau)
    np.testing.assert_allclose(prob, 0.26894142)

    tau = 0.5
    prob = sr.prob_choice_sim(cue, opt1, opt2, response, sim1, tau)
    np.testing.assert_allclose(prob, 0.11920292)


def test_choice_prob_sim2(sim1, sim2):
    """Test choice probability based on two similarity matrices."""
    cue = 0
    opt1 = 1
    opt2 = 2
    response = 0

    tau = 1
    w = 0.5
    prob = sr.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.5)

    w = 1
    prob = sr.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.26894142)

    w = 0
    prob = sr.prob_choice_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    np.testing.assert_allclose(prob, 0.731058578)


def test_induct_prob_sim(sim1, induct_trials):
    """Test induction test probability based on a similarity matrix."""
    n_trial = len(induct_trials['cue'])
    t = induct_trials
    tau = 1
    trial_prob = sr.prob_induct_sim(
        t['cue'], t['opt1'], t['opt2'], t['response'], sim1, tau
    )
    expected = np.array(
        [0.26894142, 0.73105858, 0.88079708, 0.11920292, 0.11920292, 0.88079708]
    )
    np.testing.assert_allclose(trial_prob, expected)
