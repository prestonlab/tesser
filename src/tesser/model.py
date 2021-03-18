"""Functions for fitting models of tesser behavior."""

import numpy as np
from tesser import sr
import os


def learn_sr(struct, gamma, alpha, n_state):
    """
    Train an SR matrix on the structure-learning task.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning task trials.

    gamma : float
        Discounting factor.

    alpha : float
        Learning rate.

    n_state : int
        The number of states in the environment to initialize matrices.

    Returns
    -------
    M : numpy.array
        SR Matrix for all parts and run for a given subject.
    """
    M = np.zeros([n_state, n_state])
    onehot = np.eye(n_state, dtype=np.dtype('i'))
    for (part, run), df in struct.groupby(['part', 'run']):
        envstep = df['object'].to_numpy() - 1
        envstep = envstep.astype(np.dtype('i'))
        csr.learn_sr(envstep, gamma, alpha, M, n_state, onehot)
    return M
