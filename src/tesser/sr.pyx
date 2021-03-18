cimport cython

@cython.boundscheck(False)
@cython.wraparound(False) 
def learn_sr(
    int [:] envstep,
    double gamma,
    double alpha,
    double [:,:] M,
    int n_states,
    int [:,:] onehot
):
    cdef int s
    cdef int s_new
    cdef int i
    cdef int n
    cdef double M_new
    cdef Py_ssize_t n_steps = envstep.shape[0]

    # set initial state
    s = envstep[0]
    for n in range(1, n_steps):
        s_new = envstep[n]

        # update matrix based on state transition
        for i in range(n_states):
            M_new = onehot[s_new, i] + gamma * M[s_new, i]
            M[s, i] = (1 - alpha) * M[s, i] + alpha * M_new
        s = s_new
