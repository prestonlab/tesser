import numpy as np
from tesser import sr


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


def test_choice_prob_sim1():
    """Test choice probability based on one similarity matrix."""
    sim = np.array([[1, 2, 3], [3, 2, 1], [1, 3, 2]], dtype='double')
    cue = 0
    opt1 = 1
    opt2 = 2
    response = 0

    tau = 1
    prob = sr.prob_choice_sim(cue, opt1, opt2, response, sim, tau)
    np.testing.assert_allclose(prob, 0.26894142)

    tau = 0.5
    prob = sr.prob_choice_sim(cue, opt1, opt2, response, sim, tau)
    np.testing.assert_allclose(prob, 0.11920292)


def test_choice_prob_sim2():
    """Test choice probability based on two similarity matrices."""
    sim1 = np.array([[1, 2, 3], [3, 2, 1], [1, 3, 2]], dtype='double')
    sim2 = np.array([[3, 2, 1], [1, 3, 2], [2, 1, 3]], dtype='double')
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
