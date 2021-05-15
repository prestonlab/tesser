cimport cython

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
