import numpy as np

def average_accuracy(acc_matrix, upto_task=None):
    A = np.asarray(acc_matrix, dtype=float)
    if upto_task is None: upto_task = A.shape[0] - 1
    vals = A[upto_task, :upto_task+1]
    return float(np.nanmean(vals))

def forgetting_measure(acc_matrix, upto_task=None):
    A = np.asarray(acc_matrix, dtype=float)
    if upto_task is None: upto_task = A.shape[0] - 1
    fs = []
    for j in range(upto_task):
        best = np.nanmax(A[:upto_task, j])
        final = A[upto_task, j]
        fs.append(best - final)
    return float(np.nanmean(fs)) if fs else 0.0

def backward_transfer(acc_matrix, upto_task=None):
    A = np.asarray(acc_matrix, dtype=float)
    if upto_task is None: upto_task = A.shape[0] - 1
    vals = []
    for j in range(upto_task):
        vals.append(A[upto_task, j] - A[j, j])
    return float(np.nanmean(vals)) if vals else 0.0
