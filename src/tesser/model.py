"""Functions for fitting models of tesser behavior."""

import numpy as np
import pandas as pd
from scipy import optimize
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


def create_sim(struct, n_state, alpha=None, gamma=None, sim=None):
    """Create a similarity matrix based on structure learning."""
    if sim is None:
        sim = learn_struct_sr(struct, gamma, alpha, n_state)
    return sim


def eval_dependent_param(dependent, param):
    """Evaluate dependent parameters."""
    updated = dependent.copy()
    for key, val in dependent.items():
        if isinstance(val, str):
            updated[key] = eval(val, {}, param)
    return updated


def prob_struct_induct(
    struct,
    induct,
    param,
    sim1_spec,
    sim2_spec=None,
    subject_param=None,
    question_param=None,
):
    """
    Probability of induction tests.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    param : dict
        Parameter values.

    sim1_spec : dict
        Must specify either a 'sim' field with a similarity matrix or
        'alpha' and 'gamma' to generate one from SR learning.

    sim2_spec : dict, optional
        Specification for a second similarity matrix.

    subject_param : dict, optional
        Parameters that vary by subject.

    question_param : dict, optional
        Parameters that vary by question type.

    Returns
    -------
    prob : numpy.ndarray
        Probability of the observed response for each induction test trial.
    """
    subjects = struct['subject'].unique()
    questions = induct['trial_type'].unique()
    prob = np.zeros(len(induct))
    n_state = max(
        struct['object'].max(),
        induct['cue'].max(),
        induct['opt1'].max(),
        induct['opt2'].max(),
    )
    if 'w' not in param:
        param['w'] = None
    for subject in subjects:
        subj_struct = struct.query(f'subject == {subject}')
        subj_induct = induct.query(f'subject == {subject}')
        subj_param = param.copy()
        if subject_param is not None:
            subj_param.update(subject_param[subject])

        # generate similarity matrices based on structure learning data
        sim1_spec = eval_dependent_param(sim1_spec, subj_param)
        sim1 = create_sim(subj_struct, n_state, **sim1_spec)
        if sim2_spec is not None:
            sim2_spec = eval_dependent_param(sim2_spec, subj_param)
            sim2 = create_sim(subj_struct, n_state, **sim2_spec)
        else:
            sim2 = None

        if question_param is None:
            # parameters are the same for all induction trials
            subj_prob = prob_induct(subj_induct, subj_param['tau'], sim1, subj_param['w'], sim2)
            include = induct.eval(f'subject == {subject}').to_numpy()
            prob[include] = subj_prob
        else:
            for question in questions:
                # update for question-specific parameters
                q_param = subj_param.copy()
                eval_param = eval_dependent_param(question_param[question], subj_param)
                q_param.update(eval_param)

                # evaluate the model for this question type
                q_induct = subj_induct.query(f'trial_type == {question}')
                q_prob = prob_induct(q_induct, q_param['tau'], sim1, q_param['w'], sim2)
                include = induct.eval(
                    f'subject == {subject} and trial_type == {question}'
                )
                prob[include.to_numpy()] = q_prob
    return prob


def param_bounds(var_bounds, var_names):
    """Pack group-level parameters."""
    group_lb = [var_bounds[k][0] for k in var_names]
    group_ub = [var_bounds[k][1] for k in var_names]
    bounds = optimize.Bounds(group_lb, group_ub)
    return bounds


def fit_induct(
    struct,
    induct,
    fixed,
    var_names,
    var_bounds,
    sim1_spec,
    sim2_spec=None,
    subject_param=None,
    question_param=None,
    verbose=False,
    f_optim=optimize.differential_evolution,
    optim_kws=None,
):
    """
    Fit a model of object similarity to induction data.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    fixed : dict
        Fixed parameter values.

    var_names : list
        Names of free parameters.

    var_bounds : dict
        Lower and upper limits for each free parameter.

    sim1_spec : dict
        Must specify either a 'sim' field with a similarity matrix or
        'alpha' and 'gamma' to generate one from SR learning.

    sim2_spec : dict, optional
        Specification for a second similarity matrix.

    subject_param : dict, optional
        Parameters that vary by subject.

    question_param : dict, optional
        Parameters that vary by question type.

    verbose : bool, optional
        If true, more information about the search will be displayed.

    f_optim : callable, optional
        Optimization function.

    optim_kws : dict, optional
        Keyword arguments for the optimization function.

    Returns
    -------
    logl : float
        Maximum likelihood, based on the search.

    param : dict
        Best-fitting parameter values.
    """
    if optim_kws is None:
        optim_kws = {}

    # define error function
    def f_fit(x):
        fit_param = fixed.copy()
        fit_param.update(dict(zip(var_names, x)))
        prob = prob_struct_induct(
            struct,
            induct,
            fit_param,
            sim1_spec,
            sim2_spec,
            subject_param=subject_param,
            question_param=question_param,
        )

        # handle 0 or NaN probabilities
        eps = 0.000001
        prob[np.isnan(prob)] = eps
        prob[prob < eps] = eps
        log_liklihood = np.sum(np.log(prob))
        return -log_liklihood

    # run the parameter search
    bounds = param_bounds(var_bounds, var_names)
    res = f_optim(f_fit, bounds, disp=verbose, **optim_kws)

    # fitted parameters
    param = fixed.copy()
    param.update(dict(zip(var_names, res['x'])))

    logl = -res['fun']
    return logl, param


def fit_induct_indiv(
    struct,
    induct,
    fixed,
    var_names,
    var_bounds,
    sim1_spec,
    sim2_spec=None,
    subject_param=None,
    question_param=None,
    verbose=False,
    f_optim=optimize.differential_evolution,
    optim_kws=None,
):
    """
    Fit a model of object similarity to subject induction data.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    fixed : dict
        Fixed parameter values.

    var_names : list
        Names of free parameters.

    var_bounds : dict
        Lower and upper limits for each free parameter.

    sim1_spec : dict
        Must specify either a 'sim' field with a similarity matrix or
        'alpha' and 'gamma' to generate one from SR learning.

    sim2_spec : dict, optional
        Specification for a second similarity matrix.

    subject_param : dict, optional
        Parameters that vary by subject.

    question_param : dict, optional
        Parameters that vary by question type.

    verbose : bool, optional
        If true, more information about the search will be displayed.

    f_optim : callable, optional
        Optimization function.

    optim_kws : dict, optional
        Keyword arguments for the optimization function.

    Returns
    -------
    results : pandas.DataFrame
        Search
    """
    df_list = []
    subjects = induct['subject'].unique()
    for subject in subjects:
        subj_struct = struct.query(f'subject == {subject}')
        subj_induct = induct.query(f'subject == {subject}')
        subj_param = fixed.copy()
        logl, param = fit_induct(
            subj_struct,
            subj_induct,
            subj_param,
            var_names,
            var_bounds,
            sim1_spec,
            sim2_spec=sim2_spec,
            subject_param=subject_param,
            question_param=question_param,
            verbose=verbose,
            f_optim=f_optim,
            optim_kws=optim_kws,
        )
        n = len(subj_induct)
        res = {'subject': subject, 'logl': logl, 'n': n, 'k': len(var_names)}
        res.update(param)
        df = pd.Series(res)
        df_list.append(df)
    results = pd.DataFrame(df_list)
    return results
