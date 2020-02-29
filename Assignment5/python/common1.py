import numpy as np
from scipy.ndimage import gaussian_filter
import math
from math import pi
from math import exp
import matplotlib.pyplot as plt
# Task 1a
def central_difference(I):
    """
    Computes the gradient in the u and v direction using
    a central difference filter, and returns the resulting
    gradient images (Iu, Iv) and the gradient magnitude Im.
    """
    k = np.array([0.5, 0, -0.5])
    [x,y] = np.shape(I)
    Iu = np.zeros_like(I)
    Iv = np.zeros_like(I)
    for i in range(0,y):
         Iv[:,i] = np.convolve(k,I[:,i], 'same')
    for i in range(0, x):
        Iu[i,:] = np.convolve(k, I[i,:], 'same')
    Im = np.zeros_like(I)
    for i in range(0,x):
        for j in range(0,y):
            Im[i,j] = np.sqrt(Iv[i,j]**2+Iu[i,j]**2)
    print(np.shape(Iu))
    return Iu, Iv, Im

# Task 1b
def blur(I, sigma):
    return gaussian_filter(I, sigma)
def extract_edges(Iu, Iv, Im, threshold):
    """
    Returns the u and v coordinates of pixels whose gradient
    magnitude is greater than the threshold.
    """

    # This is an acceptable solution for the task (you don't
    # need to do anything here). However, it results in thick
    # edges. If you want better results you can try to replace
    # this with a thinning algorithm as described in the text.
    v,u = np.nonzero(Im > threshold)
    theta = np.arctan2(Iv[v,u], Iu[v,u])
    return u, v, theta

def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:,:,0] + 0.5870*I[:,:,1] + 0.1140*I[:,:,2]
