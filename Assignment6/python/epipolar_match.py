import numpy as np
from util import *
def sum_squared_intensity_diff(window1,window2):
    SSI = 0
    x, y = np.shape(window2)
    for row in range(round(x/2-1)):
        for col in range(round(y/2-1)):
             SSI = SSI + (window1[row,col]-window2[row,col])**2
    return SSI

def epipolar_match(I1, I2, F, uv1):
    """
    For each point in uv1, finds the matching point in image 2 by
    an epipolar line search.

    Args:
        I1:  (H x W matrix) Grayscale image 1
        I2:  (H x W matrix) Grayscale image 2
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
        uv1: (n x 2 array) Points in image 1

    Returns:
        uv2: (n x 2 array) Best matching points in image 2.
    """

    # Tips:
    # - Image indices must always be integer.
    # - Use int(x) to convert x to an integer.
    # - Use rgb2gray to convert images to grayscale.
    # - Skip points that would result in an invalid access.
    # - Use I[v-w : v+w+1, u-w : u+w+1] to extract a window of half-width w around (v,u).
    # - Use the np.sum function.
    # todo: compute uv2
    #find matching points using sum of squared intensity differences
    # delta = sum (uv1 - uv2)**2
    uv2 = np.zeros(uv1.shape)
    w = 10
    x, y = np.shape(I1)
    print(x)
    print(y)
    #search along line for new point
    for i in range(len(uv1)):
        SSI = 1e9
        indx = np.array([uv1[i,0], uv1[i,1]])
        line = F @ np.array([uv1[i,0], uv1[i,1], 1])
        window = I1[int(uv1[i,1])-w : int(uv1[i,1])+w+1, int(uv1[i,0])-w : int(uv1[i,0])+w+1]
        for pix_x in range(int(uv1[i,0])-40,int(uv1[i,0])+40+1):
            pix_y = (-pix_x*line[0] - line[2])/line[1]
            pix_y = round(pix_y)
            if  x - w - 1 > pix_x > w - 1 and y - w - 1 > pix_y > w - 1:
                window_new = I2[int(pix_y) - w : int(pix_y) + w + 1, int(pix_x) - w : int(pix_x) + w + 1]
                temp = sum_squared_intensity_diff(window,window_new)
                if temp < SSI:
                    SSI = temp
                    indx = np.array([pix_x,pix_y])
        uv2[i,:] = indx
    print(uv1)
    print(uv2)
    plt.show()
    return uv2
