import numpy as np
from normalize_points import *

def eight_point(uv1, uv2):
    uv1, T = normalize_points(uv1)
    uv2, _ = normalize_points(uv2)
    u1 = uv1[:,0]
    v1= uv1[:,1]
    u2 = uv2[:,0]
    v2 = uv2[:,1]
    A = np.zeros((2 * len(u1), 9))
    for i in range(0, len(u1)):
        A[i,:] = np.array([u2[i]*u1[i], u2[i]*v1[i], u2[i], v2[i]*u1[i],
                 v2[i]*v1[i], v2[i], u1[i], v1[i], 1])
    U, s, v = np.linalg.svd(A) # U and v swapped positions from what is used in matlab???
    print(np.shape(v))
    F = np.reshape(v[8,:], (3, 3))
    F = T.T@F@T
    return F

def closest_fundamental_matrix(F):
    """
    Computes the closest fundamental matrix in the sense of the
    Frobenius norm. See HZ, Ch. 11.1.1 (p.280).
    """


    # todo: Compute the correct F
    return F
