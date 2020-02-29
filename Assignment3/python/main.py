import matplotlib.pyplot as plt
import numpy as np
from common import draw_frame

def estimate_H(xy, XY):
    # Task 2: Implement estimate_H
    x = xy[:,0]
    y = xy[:,1]
    X = XY[:,0]
    Y = XY[:,1]
    A = np.zeros((2 * len(X), 9))
    for i in range(0, len(X)):
        A[2 * i, :] = np.array([X[i],Y[i],1,0,0,0,-X[i]*x[i],-Y[i]*x[i],-x[i]])
        A[1+2*i,:] = np.array([0,0,0,X[i],Y[i],1,-X[i]*y[i],-Y[i]*y[i],-y[i]])
    U, s, v = np.linalg.svd(A) # U and v swapped positions from what is used in matlab???
    H= np.reshape(v[8,:], (3, 3))
    return H
def decompose_H(H):
    #
    # Task 3a: Implement decompose_H
    #
    print(H)
    H1 = H[:,0]
    H2 = H[:,1]
    H3 = np.cross(H1,H2)
    H3_conj = np.cross(-H1,-H2)

    scale1 = np.linalg.norm(H1)
    scale3 = np.linalg.norm(H3)

    Rot1 = np.zeros((3,3))
    Rot1[0:3,0] = H1/scale1
    Rot1[0:3,1] = H2/scale1
    Rot1[:,2] = H3/scale3
    Rot2 = np.zeros((3,3))
    Rot2[0:3,0] = -H1/scale1
    Rot2[0:3,1] = -H2/scale1
    Rot2[:, 2] = H3_conj/scale3

    T1 = np.eye(4) # Placeholder code
    T1[0:3,0:3] = Rot1
    T1[0:3,3] = H[:,2]/scale1
    T2 = np.eye(4)
    T2[0:3, 0:3] = Rot2
    T2[0:3, 3] = -H[:,2]/scale1
    print(T1)
    print(T2)
    return T1, T2

def choose_solution(T1, T2):
    if T1[2,3] < 0:
        return T2
    else:
        return T1

K           = np.loadtxt('../data/cameraK.txt')
all_markers = np.loadtxt('../data/markers.txt')
XY          = np.loadtxt('../data/model.txt')
n           = len(XY)

for image_number in range(0,1):
    image_number = 5
    I = plt.imread('../data/video%04d.jpg' % image_number)
    markers = all_markers[image_number,:]
    markers = np.reshape(markers, [n, 3])
    matched = markers[:,0].astype(bool) # First column is 1 if marker was detected
    uv = markers[matched, 1:3] # Get markers for which matched = 1

    # Convert pixel coordinates to normalized image coordinates
    xy = (uv - K[0:2,2])/np.array([K[0,0], K[1,1]])
    H = estimate_H(xy, XY[matched, :2])
    T1,T2 = decompose_H(H)
    T = choose_solution(T1, T2)

    # Compute predicted corner locations using model and homography
    uv_hat = (K@H@XY.T)
    uv_hat = (uv_hat/uv_hat[2,:]).T
    plt.clf()
    plt.imshow(I, interpolation='bilinear')
    draw_frame(K, T, scale=7)
    plt.scatter(uv[:,0], uv[:,1], color='red', label='Observed')
    plt.scatter(uv_hat[:,0], uv_hat[:,1], marker='+', color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.title(' coordinate frame from H on image 0005')
    plt.show()
    plt.savefig('../data/out%04d.png' % image_number)
