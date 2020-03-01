import numpy as np
from linear_triangulation import *
from camera_matrices import *

def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    args:
    K1: camera 1 intrinsics
    K2: camera 2 intrinsics
    Rts: possible rotations and translations from cam1 to cam2
    """
    # todo: Choose the correct solution
    # correct solution gives projected point with positive Z in both frames
    X = np.zeros((len(Rts),4*len(uv1)))
    for i in range(len(Rts)):
        pose = Rts[i]
        p_1,p_2 = camera_matrices(K1, K2, pose[0], pose[1])
        for j in range(len(uv1)):
            X[i, 4*j : 4*(j+1)] = np.squeeze(linear_triangulation(uv1[j], uv2[j], p_1, p_2))
    count = np.zeros((4,1))
    for i in range(len(Rts)):
        for j in range(len(uv1)):
            if X[i,3*j+2] > 0:
                count[i] += 1
    print(count)
    soln = np.argmax(count)
    print('Choosing solution %d' % soln)
    return Rts[soln]
