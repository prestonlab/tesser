"""Functions for fitting models of tesser behavior."""

import numpy as np
import pandas as pd
from scipy import optimize
from joblib import Parallel, delayed
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
    # export induction data for use with cython code
    induct = induct.reset_index()
    cue = induct['cue'].to_numpy().astype(np.dtype('i')) - 1
    opt1 = induct['opt1'].to_numpy().astype(np.dtype('i')) - 1
    opt2 = induct['opt2'].to_numpy().astype(np.dtype('i')) - 1
    response = induct['response'].to_numpy().astype(np.dtype('i')) - 1

    # for trials with no response, place a temporary dummy response
    missing = induct['response'].isna().to_numpy()
    response[missing] = 0
    if sim2 is None:
        prob = learn.prob_induct_sim(cue, opt1, opt2, response, sim1, tau)
    else:
        if w is None:
            raise ValueError('If sim2 is defined, w must be defined.')
        prob = learn.prob_induct_sim2(cue, opt1, opt2, response, sim1, sim2, w, tau)

    # probability of no response is undefined
    prob[missing] = np.nan
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
    questions = induct['trial_type'].cat.categories.values
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
        subj_sim1_spec = eval_dependent_param(sim1_spec, subj_param)
        sim1 = create_sim(subj_struct, n_state, **subj_sim1_spec)
        if sim2_spec is not None:
            subj_sim2_spec = eval_dependent_param(sim2_spec, subj_param)
            sim2 = create_sim(subj_struct, n_state, **subj_sim2_spec)
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
                q_induct = subj_induct.query(f'trial_type == "{question}"')
                q_prob = prob_induct(q_induct, q_param['tau'], sim1, q_param['w'], sim2)
                include = induct.eval(
                    f'subject == {subject} and trial_type == "{question}"'
                )
                prob[include.to_numpy()] = q_prob
    return prob


def param_bounds(var_bounds, var_names):
    """Pack group-level parameters."""
    group_lb = [var_bounds[k][0] for k in var_names]
    group_ub = [var_bounds[k][1] for k in var_names]
    bounds = optimize.Bounds(group_lb, group_ub)
    return bounds


def _fit_induct(
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


def _fit_subject(struct, induct, subject, fixed, var_names, *args, **kwargs):
    """Run a parameter search for one subject."""
    subj_struct = struct.query(f'subject == {subject}')
    subj_induct = induct.query(f'subject == {subject}')
    subj_param = fixed.copy()
    logl, param = _fit_induct(
        subj_struct, subj_induct, subj_param, var_names, *args, **kwargs
    )
    n = len(subj_induct)
    res = {'logl': logl, 'n': n, 'k': len(var_names)}
    res.update(param)
    s = pd.Series(res)
    return s


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
    n_rep=1,
    n_job=1,
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

    n_rep : int, optional
        Number of times to repeat the search.

    n_job : int, optional
        Number of jobs to run in parallel.

    verbose : bool, optional
        If true, more information about the search will be displayed.

    f_optim : callable, optional
        Optimization function.

    optim_kws : dict, optional
        Keyword arguments for the optimization function.

    Returns
    -------
    results : pandas.Series
        Search results, including log-likelihood (logl), the number of
        data points fit (n), the number of free parameters (k), and the
        best-fitting value of each parameter.
    """
    full_results = Parallel(n_jobs=n_job)(
        delayed(_fit_induct)(
            struct,
            induct,
            fixed,
            var_names,
            var_bounds,
            sim1_spec,
            sim2_spec=sim2_spec,
            subject_param=subject_param,
            question_param=question_param,
            verbose=verbose,
            f_optim=f_optim,
            optim_kws=optim_kws,
        ) for _ in range(n_rep)
    )
    n = len(induct)
    k = len(var_names)
    d = {
        rep: {
            'logl': logl, 'n': n, 'k': k, **param
        } for rep, (logl, param) in zip(range(n_rep), full_results)
    }
    results = pd.DataFrame(d).T
    results = results.astype({'n': int, 'k': int})
    results.index.rename('rep', inplace=True)
    return results


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
    n_rep=1,
    n_job=1,
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

    n_rep : int, optional
        Number of times to repeat the search.

    n_job : int, optional
        Number of jobs to run in parallel.

    verbose : bool, optional
        If true, more information about the search will be displayed.

    f_optim : callable, optional
        Optimization function.

    optim_kws : dict, optional
        Keyword arguments for the optimization function.

    Returns
    -------
    results : pandas.DataFrame
        Search results for each participant, including log-likelihood
        (logl), the number of data points fit (n), the number of free
        parameters (k), and the best-fitting parameter value of each
        parameter.
    """
    subjects = induct['subject'].unique()
    full_subjects = np.repeat(subjects, n_rep)
    full_reps = np.tile(np.arange(n_rep), len(subjects))
    full_results = Parallel(n_jobs=n_job)(
        delayed(_fit_subject)(
            struct,
            induct,
            subject,
            fixed,
            var_names,
            var_bounds,
            sim1_spec,
            sim2_spec=sim2_spec,
            subject_param=subject_param,
            question_param=question_param,
            verbose=verbose,
            f_optim=f_optim,
            optim_kws=optim_kws,
        ) for subject in full_subjects
    )
    d = {(subject, rep): res for subject, rep, res in
         zip(full_subjects, full_reps, full_results)}
    results = pd.DataFrame(d).T
    results.index.rename(['subject', 'rep'], inplace=True)
    results = results.astype({'n': int, 'k': int})
    return results


def fit_induct_question(struct, induct, *args, **kwargs):
    """
    Fit induction data separately by question type.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    See fit_induct for other parameters.

    Returns
    -------
    results : pandas.DataFrame
        Search results for each question type and participant.
    """
    questions = induct['trial_type'].cat.categories.values
    res_list = []
    for question in questions:
        induct_question = induct.query(f'trial_type == "{question}"')
        res = fit_induct(struct, induct_question, *args, **kwargs)
        res_list.append(res)
    results = pd.concat(res_list, axis=0, keys=questions)
    results.index.rename(['trial_type', 'rep'], inplace=True)
    return results


def fit_induct_indiv_question(struct, induct, *args, **kwargs):
    """
    Fit induction data separately by subject and question type.

    Parameters
    ----------
    struct : pandas.DataFrame
        Structure learning data.

    induct : pandas.DataFrame
        Induction test data.

    See fit_induct_indiv for other parameters.

    Returns
    -------
    results : pandas.DataFrame
        Search results for each question type and participant.
    """
    questions = induct['trial_type'].cat.categories.values
    res_list = []
    for question in questions:
        induct_question = induct.query(f'trial_type == "{question}"')
        res = fit_induct_indiv(struct, induct_question, *args, **kwargs)
        res_list.append(res)
    results = pd.concat(res_list, axis=0, keys=questions)
    results.index.rename(['trial_type', 'subject', 'rep'], inplace=True)
    return results


def get_best_results(results):
    """Get best results from a repeated search."""
    if isinstance(results.index, pd.MultiIndex):
        groups = results.index.names[:-1]
        df = []
        for ind, res in results.groupby(groups):
            rep = res['logl'].argmax()
            df.append(res.loc[[(ind, rep)]])
        best = pd.concat(df, axis=0)
    else:
        best = results.loc[[results['logl'].argmax()]]
    return best


def get_correct_prob(stats):
    """Get the probability of correct response."""
    stats = stats.copy()
    stats['prob_correct'] = np.nan
    corr = stats['correct'] == 1
    incorr = stats['correct'] == 0

    # when the response was correct, P(correct) = P(response)
    stats.loc[corr, 'prob_correct'] = stats.loc[corr, 'prob_response']

    # when the response was incorrect, P(correct) = 1 - P(response)
    stats.loc[incorr, 'prob_correct'] = 1 - stats.loc[incorr, 'prob_response']
    return stats


def get_fitted_prob(results, induct, struct, *args, **kwargs):
    """Get fitted probability for each trial."""
    if not isinstance(results.index, pd.MultiIndex):
        groups = False
    else:
        groups = True
    i = results.index.names.index('rep')
    results = results.reset_index(i)

    stats = induct.copy()
    if not groups:
        # no groups to deal with; can evaluate in one step
        param = results.loc[0].to_dict()
        prob = prob_struct_induct(struct, induct, param, *args, **kwargs)
        stats['prob_response'] = prob
        stats = get_correct_prob(stats)
        return stats

    # calculate trial probabilities for each group
    names = results.index.names
    for ind, res in results.iterrows():
        # get relevant trials for this group
        inc_struct = np.ones(len(struct), dtype=bool)
        inc_induct = np.ones(len(induct), dtype=bool)
        for name, val in zip(names, ind):
            if name == 'subject':
                inc_struct &= struct[name] == val
            if name in induct.columns:
                inc_induct &= induct[name] == val

        # evaluate trial probabilities
        param = res.to_dict()
        prob = prob_struct_induct(
            struct[inc_struct], induct[inc_induct], param, *args, **kwargs
        )
        stats.loc[inc_induct, 'prob_response'] = prob
        stats = get_correct_prob(stats)
    return stats
