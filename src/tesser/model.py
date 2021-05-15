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

    Returns
    -------
    prob : numpy.ndarray
        The probability of each induction test trial.
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


def prob_struct_induct(
    struct, induct, alpha, gamma, tau, w=None, alpha2=None, gamma2=None, sim=None
):
    """
    Calculate response probabilities for induction given structure learning.

    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    alpha : float
        Learning rate during structure learning.

    gamma : float
        Discounting factor.

    tau : float
        Temperature parameter for softmax choice rule.

    w : float, optional
        Weighting of the sim1 matrix relative to sim2.

    alpha2 : float, optional
        Learning rate for second SR during structure lrearning.

    gamma2 : float, optional
        Dicounding factor for second SR.

    sim : numpy.ndarray, optional
        [item x item] array with the similarity of each item pair.

    Returns
    -------
    prob : numpy.ndarray
        The probability of each induction test trial.
    """
    # learn an SR representation from the structure data
    n_state = max(
        struct['object'].max(),
        induct['cue'].max(),
        induct['opt1'].max(),
        induct['opt2'].max(),
    )
    SR = learn_struct_sr(struct, gamma, alpha, n_state)

    # calculate induction trial probabilities
    if alpha2 is not None and gamma2 is not None:
        # second SR representation
        SR2 = learn_struct_sr(struct, gamma2, alpha2, n_state)
        prob = prob_induct(induct, tau, SR, w=w, sim2=SR2)
    elif sim is not None:
        # some similarity matrix that doesn't depend on structure learning
        prob = prob_induct(induct, tau, SR, w=w, sim2=sim)
    else:
        # just one similarity matrix based on SR
        prob = prob_induct(induct, tau, SR)
    return prob
