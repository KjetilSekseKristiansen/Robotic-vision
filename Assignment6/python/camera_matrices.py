import numpy as np

def camera_matrices(K1, K2, R, t):
    """ Computes the projection matrix for camera 1 and camera 2.

    Args:
        K1,K2: Intrinsic matrix for camera 1 and camera 2.
        R,t: The rotation and translation mapping points in camera 1 to points in camera 2.

    Returns:
        P1,P2: The projection matrices with shape 3x4.
    """
    #choosing frame 1 as reference frame
    pose_ref = np.zeros((3,4))
    pose_ref[0:3,0:3] = np.identity(3)

    pose_mat = np.zeros((3,4))
    pose_mat[0:3,0:3] = R
    pose_mat[0:3,3] = t

    P1 = K1 @ pose_ref
    P2 = K2 @ pose_mat
    return P1, P2
