import numpy as np

def essential_from_fundamental(F, K1, K2):
    """""
    Args:
        F:  Fundamental Matrix
        K1: Intrinsic matrix for camera 1
        K2: Intrinsic matrix for camera 2
    """
    E = K2.T@F@K1
    return E
