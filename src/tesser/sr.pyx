cimport cython
from libc.math cimport exp

@cython.boundscheck(False)
@cython.wraparound(False) 
def learn_sr(
    double [:,:] M,
    int [:] envstep,
    double gamma,
    double alpha,
):
    cdef int s
    cdef int s_new
    cdef int i
    cdef int n
    cdef double M_new
    cdef double hot
    cdef Py_ssize_t n_states = M.shape[0]
    cdef Py_ssize_t n_steps = envstep.shape[0]

    # set initial state
    s = envstep[0]
    for n in range(1, n_steps):
        s_new = envstep[n]

        # update matrix based on state transition
        for i in range(n_states):
            if s_new == i:
                hot = 1
            else:
                hot = 0
            M_new = hot + gamma * M[s_new, i]
            M[s, i] = (1 - alpha) * M[s, i] + alpha * M_new
        s = s_new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef prob_choice_sim1(
    int cue,
    int opt1,
    int opt2,
    int response,
    double [:,:] sim,
    double tau,
):
    """Choice probability given one similarity matrix."""
    cdef double support1
    cdef double support2
    cdef double support
    cdef double prob
    support1 = sim[cue, opt1]
    support2 = sim[cue, opt2]
    if response == 0:
        support = support1
    else:
        support = support2
    prob = exp(support / tau) / (exp(support1 / tau) + exp(support2 / tau))
    return prob
