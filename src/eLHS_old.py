from scipy.stats.qmc import LatinHypercube as LHSampler
from matplotlib import pyplot as plt
import numpy as np
import src.utils as utils

## ----------------------------------------------------------------
## --------------------- EXPANSION ALGO ---------------------------
## ----------------------------------------------------------------
def eLHS(lhs: np.ndarray, M:int, fullGraded = True):
    N, P = lhs.shape
    if(fullGraded and grade(lhs) < 1.0):
        raise ValueError("Parameter `lhs` must be a proper full-graded LHS sample set.")
    
    # returning container
    exp = {"F": None,
           "M": M,
           "Q": None,
           "N": N,
           "expansion": None, 
           "grade": None }
    
    mExpansion = sow(lhs, M)
    eLHS = utils.concat(lhs, mExpansion)
    eGrade = grade(eLHS)

    devFlag = True
    if(devFlag or eGrade == N + M):
        exp["F"] = exp["Q"] = M
        exp["expansion"] = mExpansion
        exp["grade"] = eGrade
    else:
        pass
    
    return exp


## ----------------------------------------------------------------
## ---------------------- LSSP Scaling ----------------------------
## ----------------------------------------------------------------
def scale_down(eLHS: np.ndarray, N: int, M: int):
    pass

def scale_up(eLHS: np.ndarray, N: int, M: int):
    pass


## ----------------------------------------------------------------
## --------------------- GRADING SYSTEM ---------------------------
## ----------------------------------------------------------------

def trace(PeLHS: np.ndarray):
    ''' 
        Returns auxiliary H(qij) variable that traces the presence of each 
        sample Xij in each q-th interval of the j-th dimension.
    '''
    N, P = PeLHS.shape
    Q = N   # for sake of clarity
    h = np.zeros((Q, N, P), dtype=int)
    for q in range(Q):
        for i in range(N):
            for j in range(P):
                if(PeLHS[i, j] is not None and not np.isnan(PeLHS[i, j]) and
                   q/Q <= PeLHS[i, j] <= (q+1)/Q):
                    h[q, i, j] = int(1)
    return h

def count_fingerprints(lhs: np.ndarray, verbose = True):
    '''
        Return auxiliary Yqj variable that counts the number of occurrences of 
        each sample Xij in q-th interval of the j-th dimension.
        The verbose flag could be used to return a binary version of Yqj 
        whose values are { 1 if there is at least one sample Xij in qj-th 
        interval, 0 otherwise }
    '''
    h = trace(lhs)
    Q, _, P = h.shape
    y = np.zeros((Q, P), dtype=int)
    for q in range(Q):
        for j in range(P):
            s = np.sum(h[q, :, j])
            y[q, j] = (s if verbose else (1 if s > 0 else 0) )
    return y

def section_complexity(lhs, n, j):
    return np.sum(
        [ 
            1 if np.any(utils.H(lhs[:, j] - q/n) * utils.H((q+1)/n - lhs[:, j]) > 0, axis=None) else 0
        for q in range(n) ]
    )


def grade(lhs: np.ndarray, n = None, mode = 3):
    '''
        TO UPDATE:
        Returns the sum-reduction y to a single value that
        represents the quality of the lhs.
    '''
    N, P = lhs.shape
    y = count_fingerprints(lhs, verbose=False)
    # reducing y
    if mode == 1:
        return np.sum([np.sum(y[:, s]) for s in range(y.shape[1])]) / (N * P)
    elif mode == 2:
        return np.min([np.sum(y[:, s]) for s in range(y.shape[1])]) / N
    elif mode == 3:
        n = n if n is not None else N   # n = n??N
        return np.sum([
            section_complexity(lhs, n, j) for j in range(P)
        ]) / (n * P)
    elif mode == 4:
        n = n if n is not None else N   # n = n??N
        return [section_complexity(lhs, n, j) for j in range(P)]

## ----------------------------------------------------------------
## ------------------------ ROUTINES ------------------------------
## ----------------------------------------------------------------
# empty-extend the LHS sample set NxP to a (N+M)xP set
def empty_expansion(lhs: np.ndarray, M:int):
    _, P = lhs.shape
    PeLHS = np.concatenate((lhs, [[None for _ in range(P)] for _ in range(M)]))
    return PeLHS


def sow(lhs: np.ndarray, M: int, scattering = False):
    N, P = lhs.shape
    LHS = LHSampler(d = P)  # should be parametrized (sharing one instance of LHS)

    PeLHS = empty_expansion(lhs, M)
    newSamples = np.zeros((M, P))
    sdk = count_fingerprints(PeLHS)

    # list of available dim-vacancies in PeLHS
    vacancies = [
        [i for i in range(N + M) if sdk[i, j] == 0]
        for j in range(P)
    ]

    expansion = LHS.random(M)
    timestep = 1/(N + M)
    # r.shuffle(expansion)
    for m in range(M):
        samplex = utils.inner_coords(expansion[m], M)
        for j in range(P):
            q, vacancies[j] = utils.rpop(vacancies[j])
            newSamples[m, j] = (q + samplex[j]) * timestep
    return newSamples
