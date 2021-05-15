"""Functions for fitting models of tesser behavior."""

import numpy as np
from tesser import learn


def learn_struct_sr(struct, gamma, alpha, n_state):
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
    for _, df in struct.groupby(['part', 'run']):
        envstep = df['object'].to_numpy() - 1
        envstep = envstep.astype(np.dtype('i'))
        learn.learn_sr(M, envstep, gamma, alpha)
    return M


def prob_induct(induct, tau, sim1, w=None, sim2=None):
    """
    Calculate response probabilities for induction tests.

    Parameters
    ----------
    induct : pandas.DataFrame
        Induction test data.

    tau : float
        Temperature parameter for softmax choice rule.

    sim1 : numpy.ndarray
        [item x item] array with the similarity of each item pair.

    w : float, optional
        Weighting of the sim1 matrix relative to sim2.

    sim2 : numpy.ndarray, optional
        [item x item] array with the similarity of each item pair.
    """
    induct = induct.reset_index()
    cue = induct['cue'].to_numpy().astype(np.dtype('i')) - 1
    opt1 = induct['opt1'].to_numpy().astype(np.dtype('i')) - 1
    opt2 = induct['opt2'].to_numpy().astype(np.dtype('i')) - 1
    response = induct['response'].to_numpy().astype(np.dtype('i')) - 1
    if sim2 is None:
        prob = learn.prob_induct_sim(cue, opt1, opt2, response, sim1, tau)
    else:
        if w is None:
            raise ValueError('If sim2 is defined, w must be defined.')
        prob = learn.prob_induct_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)
    return prob
