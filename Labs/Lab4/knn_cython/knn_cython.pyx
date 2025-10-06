import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Declare numpy types for Cython
DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t LTYPE_t

def KNN_algorithm(np.ndarray[DTYPE_t, ndim=2] x_train,
                  np.ndarray[DTYPE_t, ndim=2] x_test,
                  np.ndarray[LTYPE_t, ndim=1] class_train,
                  int n_neighbours):
    """
    Cython-optimized version of the KNN algorithm.
    """

    cdef int N_train = x_train.shape[0]
    cdef int N_test = x_test.shape[0]
    cdef int dim = x_train.shape[1]

    # Create numpy arrays (not Python lists!)
    cdef np.ndarray[DTYPE_t, ndim=2] x_test_dist = np.zeros((N_test, N_train), dtype=np.float64)

    # Memory views for fast access
    cdef double[:, :] x_train_mv = x_train
    cdef double[:, :] x_test_mv = x_test
    cdef double[:, :] x_test_dist_mv = x_test_dist

    cdef int i, j, k
    cdef double diff, dist

    # --- Compute distances manually (this is the slow part in Python) ---
    for i in range(N_test):
        for j in range(N_train):
            dist = 0.0
            for k in range(dim):
                diff = x_test_mv[i, k] - x_train_mv[j, k]
                dist += diff * diff
            x_test_dist_mv[i, j] = sqrt(dist)

    # Convert back to numpy array to use NumPy’s sorting and bincount (still fine)
    x_test_dist = np.asarray(x_test_dist_mv)
    x_test_dist_ordered_id = np.argsort(x_test_dist, axis=1)

    # --- Get labels of closest neighbours ---
    x_test_neighbours_label = []
    for i in range(N_test):
        idx = x_test_dist_ordered_id[i][:n_neighbours]
        labels = class_train[idx]
        x_test_neighbours_label.append(labels)

    x_test_neighbours_label = np.array(x_test_neighbours_label).astype(np.int64)

    # --- Predict class ---
    class_pred = []
    for i in range(N_test):
        t = np.bincount(x_test_neighbours_label[i])
        pred = np.argmax(t)
        class_pred.append(int(pred))

    return class_pred
