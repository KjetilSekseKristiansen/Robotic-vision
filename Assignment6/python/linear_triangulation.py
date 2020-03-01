import numpy as np

def linear_triangulation(uv1, uv2, P1, P2):
    """
    Compute the 3D position of a single point from 2D correspondences.

    Args:
        uv1:    2D projection of point in image 1.
        uv2:    2D projection of point in image 2.
        P1:     Projection matrix with shape 3 x 4 for image 1.
        P2:     Projection matrix with shape 3 x 4 for image 2.

    Returns:
        X:      3D coordinates of point in the camera frame of image 1.
                (not homogeneous!)

    See HZ Ch. 12.2: Linear triangulation methods (p312)
    """
    u1 = uv1[0]
    v1 = uv1[1]
    u2 = uv2[0]
    v2 = uv2[1]
    A = np.zeros((4,4))
    A[0] = np.reshape(u1*P1[2,:] - P1[0,:],(1,4))
    A[1] = np.reshape(v1*P1[2,:] - P1[1,:],(1,4))
    A[2] = np.reshape(u2 * P2[2, :] - P2[0, :],(1,4))
    A[3] = np.reshape(v2 * P2[2, :] - P2[1, :],(1,4))
    _, _, v = np.linalg.svd(A)
    X = np.squeeze(v[-1,:])
    return X
