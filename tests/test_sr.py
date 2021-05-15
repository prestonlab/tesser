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
