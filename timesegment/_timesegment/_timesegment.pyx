#cython: boundscheck=False, cdivision=True, wraparound=False

import numpy as np
cimport numpy as np

cimport cython

####################################
# Type Declarations
####################################

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef struct vector_min:
    int mini
    np.float_t minn

ctypedef vector_min vector_min_t

cdef struct fit_ret:
    float alpha
    float beta
    float mean_squared_error

ctypedef fit_ret fit_ret_t

cdef struct best_split_ret:
    int splittingIndex
    float min_lrmse
    float leftalpha
    float leftbeta
    float leftmse
    float rightalpha
    float rightbeta
    float rightmse

ctypedef best_split_ret best_split_ret_t

####################################
# cdef Functions
####################################

cdef vector_min_t get_min_and_i(np.ndarray[np.float_t, ndim=1] vector):
    cdef int nb_sample
    cdef int i
    cdef vector_min_t res

    res.mini = 0
    res.minn = 0.0
    res.minn = vector[0]

    nb_sample = vector.shape[0]

    for i in range(1, nb_sample):
        if vector[i] < res.minn:
            res.minn = vector[i]
            res.mini = i
    return res

cdef float mean(np.ndarray[np.float_t, ndim=1] vector):
    cdef float res
    cdef int nb_sample, i
    res = 0.0
    nb_sample = vector.shape[0]
    for i in range(nb_sample):
        res += vector[i]
    return res/nb_sample


# Both vector should have the same size
cdef float cov(np.ndarray[np.float_t, ndim=1] vector1, np.ndarray[np.float_t, ndim=1] vector2):
    cdef float mean1
    cdef float mean2
    cdef float term1
    cdef int nb_sample
    cdef int i

    mean1 = mean(vector1)
    mean2 = mean(vector2)
    term1 = 0.0
    nb_sample = vector1.shape[0]

    for i in range(nb_sample):
        term1 += vector1[i] * vector2[i]

    return 1.0/nb_sample * (term1 - 1.0*nb_sample*mean1*mean2)

# Both vector should have the same size
cdef float var(np.ndarray[np.float_t, ndim=1] vector):
    cdef float meann
    cdef float term1
    cdef int nb_sample
    cdef float ret
    cdef int i

    meann = mean(vector)
    term1 = 0.0
    nb_sample = vector.shape[0]

    for i in range(nb_sample):
        term1 += vector[i]**2

    return 1.0 / nb_sample * (term1 - 1.0*nb_sample * meann**2)

####################################
# cpdef Functions
####################################

cpdef best_split_ret_t best_split(np.ndarray[np.float_t, ndim=1] data, int tau):
    cdef int numObs
    cdef int splittingIndex
    cdef float splittingIndex_f
    cdef float min_lrmse
    cdef int t

    cdef best_split_ret_t best_split_res
    cdef fit_ret_t fit_res_l
    cdef fit_ret_t fit_res_r

    numObs = data.shape[0]

    leftmse = np.zeros((numObs-2*tau+1,), dtype = float)
    leftalpha = np.zeros((numObs-2*tau+1,), dtype = float)
    leftbeta = np.zeros((numObs-2*tau+1,), dtype = float)
    rightmse = np.zeros((numObs-2*tau+1,), dtype = float)
    rightalpha = np.zeros((numObs-2*tau+1,), dtype = float)
    rightbeta = np.zeros((numObs-2*tau+1,), dtype = float)
    lr_mse = np.zeros((numObs-2*tau+1,), dtype = float)

    for t in range(tau, numObs-tau+1):
        fit_res_l = fit(data[:t])
        fit_res_r = fit(data[t:])
        leftalpha[t - tau] = fit_res_l.alpha
        leftbeta[t - tau] = fit_res_l.beta
        leftmse[t - tau] = fit_res_l.mean_squared_error
        rightalpha[t - tau] = fit_res_r.alpha
        rightbeta[t - tau] = fit_res_r.beta
        rightmse[t - tau] = fit_res_r.mean_squared_error
        lr_mse[t - tau] = fit_res_l.mean_squared_error + fit_res_r.mean_squared_error

    res_min = get_min_and_i(lr_mse)

    splittingIndex = res_min.mini
    min_lrmse = res_min.minn

    best_split_res.splittingIndex = res_min.mini+tau
    best_split_res.min_lrmse = res_min.minn
    best_split_res.leftalpha = leftalpha[res_min.mini]
    best_split_res.leftbeta = leftbeta[res_min.mini]
    best_split_res.leftmse = leftmse[res_min.mini]
    best_split_res.rightalpha = rightalpha[res_min.mini]
    best_split_res.rightbeta = rightbeta[res_min.mini]
    best_split_res.rightmse = rightmse[res_min.mini]

    return best_split_res

cpdef np.ndarray get_preds(np.ndarray[np.float_t, ndim=1] data, float alpha, float beta):
    cdef int numObs
    cdef int i

    numObs = data.shape[0]
    ret = np.zeros((numObs,), dtype = float)
    predictors = np.arange(1, numObs + 1, dtype = float)

    for i in range(numObs):
        ret[i] = alpha + beta * predictors[i]

    return ret

cpdef fit_ret_t fit(np.ndarray[np.float_t, ndim=1] data):
    cdef int numObs
    cdef float cov_pred_data
    cdef float var_pred
    cdef float var_data
    cdef fit_ret_t ret

    numObs = data.shape[0]
    if numObs == 1:
        ret.alpha = data[0]
        ret.beta = 0.0
        ret.mean_squared_error = 0.0
        return ret

    predictors = np.arange(1, numObs + 1, dtype = float)

    cov_pred_data = cov(predictors, data)
    var_pred = var(predictors)
    var_data = var(data)

    ret.beta = cov_pred_data/var_pred
    ret.alpha = mean(data) - ret.beta*mean(predictors)
    ret.mean_squared_error = var_data - (cov_pred_data**2)/var_pred

    return ret

####################################
# Classes
####################################

class Partition_tree:
    def __init__(self, np.ndarray[np.float_t, ndim=1] values, int max_depth, int nb_segments, float delta_complexity, int tau):
        assert values.dtype == DTYPE

        cdef fit_ret_t fit_res

        self.signal = values  # supposed to be numpy array
        self.max_depth = max_depth
        self.nb_segments = nb_segments
        self.delta_complexity = delta_complexity
        self.tau = tau

        # Tree structure
        fit_res = fit(self.signal)
        self.head = Partition_node(self.signal, self.max_depth, self.nb_segments, self.delta_complexity, tau, fit_res.alpha, fit_res.beta,
                                   fit_res.mean_squared_error, self.delta_complexity)

    def split(self):
            self.head.split()

    def get_predictions(self):
        return self.head.get_predictions()

    def get_current_nb_segment(self):
        return self.head.get_nb_segments()

    def weakest_link_pruning(self):
        while self.get_current_nb_segment() > self.nb_segments:
            to_delete = self.head.get_weakest_link()
            to_delete.delete_children()

    def get_durations(self):
        return self.head.get_durations()


cdef class Partition_node:
    cdef np.ndarray signal
    cdef int max_depth
    cdef int nb_segments
    cdef float delta_complexity
    cdef int current_depth
    cdef int numObs
    cdef float outSlope
    cdef float outOrigin
    cdef float outMSE
    cdef int outDuration
    cdef float mse_improvement
    cdef Partition_node left_child
    cdef Partition_node right_child
    cdef np.ndarray outPrediction
    cdef int tau

    def __cinit__(self, np.ndarray[np.float_t, ndim=1] signal, int max_depth, int nb_segments, float delta_complexity, int tau, float alpha, float beta, float mse, int current_depth=0):
        assert signal.dtype == np.float

        self.signal = signal
        self.max_depth = max_depth
        self.nb_segments = nb_segments
        self.delta_complexity = delta_complexity

        # Tree structure
        self.left_child = None
        self.right_child = None

        self.current_depth = current_depth
        self.numObs = self.signal.shape[0]
        self.outSlope = beta  # slopes
        self.outOrigin = alpha  # origins
        self.outMSE = mse  # MSE
        self.outDuration = self.signal.shape[0]
        self.outPrediction = np.zeros((self.numObs, ), dtype = float)  # predictions
        self.mse_improvement = 0.0
        self.tau = tau


    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def split(self):

        cdef float minMse, leftAlpha, leftBeta, leftMse, rightAlpha, rightBeta, rightMse
        cdef int splittingIndex_i

        cdef best_split_ret_t res

        if self.max_depth > -1 and self.current_depth >= self.max_depth:
            return
        if self.outDuration < 2:
            return
        if not self.outMSE != 0.0:
            return
        if self.outDuration < 2*self.tau:
            return

        res = best_split(self.signal, self.tau)

        splittingIndex_i = res.splittingIndex
        minMse = res.min_lrmse
        leftAlpha = res.leftalpha
        leftBeta = res.leftbeta
        leftMse = res.leftmse
        rightAlpha = res.rightalpha
        rightBeta = res.rightbeta
        rightMse = res.rightmse

        # If no improvement, return
        if minMse >= self.outMSE:
            return

        if np.abs(self.outMSE - minMse) / self.outMSE <= self.delta_complexity:
            return

        self.mse_improvement = self.outMSE - minMse

        self.left_child = Partition_node(self.signal[:splittingIndex_i],
                                         self.max_depth,
                                         self.nb_segments,
                                         self.delta_complexity,
                                         self.tau,
                                         leftAlpha,
                                         leftBeta,
                                         leftMse,
                                         self.current_depth + 1)
        self.right_child = Partition_node(self.signal[splittingIndex_i:],
                                          self.max_depth,
                                          self.nb_segments,
                                          self.delta_complexity,
                                          self.tau,
                                          rightAlpha,
                                          rightBeta,
                                          rightMse,
                                          self.current_depth + 1)
        [child.split() for child in iter([self.left_child, self.right_child])]

        return

    def get_predictions(self):
        if self.left_child is not None and self.right_child is not None:
            return np.append(self.left_child.get_predictions(), self.right_child.get_predictions())
        else:
            return get_preds(self.signal, self.outOrigin, self.outSlope)

    def get_nb_segments(self):
        if self.left_child is None and self.right_child is None:
            return 1
        else:
            return self.left_child.get_nb_segments() + self.right_child.get_nb_segments()

    def get_durations(self):
        if self.left_child is None and self.right_child is None:
            return self.outDuration
        else:
            # TODO : Change np.append in somethong not numpy
            return np.append(self.left_child.get_durations(), self.right_child.get_durations())

    def is_terminal(self):
        return self.left_child is None and self.right_child is None

    def is_last_split(self):
        return (self.left_child.left_child is None and
                self.left_child.right_child is None and
                self.right_child.left_child is None and
                self.right_child.right_child is None)

    def get_weakest_link(self):
        if self.is_last_split():
            return self
        else:
            if self.left_child.is_terminal():
                return self.right_child.get_weakest_link()
            elif self.right_child.is_terminal():
                return self.left_child.get_weakest_link()
            else:
                left_weakest = self.left_child.get_weakest_link()
                right_weakest = self.right_child.get_weakest_link()
                # if left_weakest.mse_improvement < right_weakest.mse_improvement:
                if left_weakest.get_mse_improvement() < right_weakest.get_mse_improvement():
                    return left_weakest
                else:
                    return right_weakest

    def delete_children(self):
        self.left_child = None
        self.right_child = None

    def get_mse_improvement(self):
        return self.mse_improvement
