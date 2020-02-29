import numpy as np
import sympy as sp
from math import pi
K                  = np.loadtxt('../data/cameraK.txt')
p_model            = np.loadtxt('../data/model.txt')
platform_to_camera = np.loadtxt('../data/pose.txt')

def residuals(uv, weights, yaw, pitch, roll):

    # Helicopter model from Exercise 1 (you don't need to modify this).
    base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
    hinge_to_base    = translate(0, 0, 0.325)@rotate_y(pitch)
    arm_to_hinge     = translate(0, 0, -0.0552)
    rotors_to_arm    = translate(0.653, 0, -0.0312)@rotate_x(roll)
    base_to_camera   = platform_to_camera@base_to_platform
    hinge_to_camera  = base_to_camera@hinge_to_base
    arm_to_camera    = hinge_to_camera@arm_to_hinge
    rotors_to_camera = arm_to_camera@rotors_to_arm
    #
    # Task 1a: Implement the rest of this function
    coord = np.zeros((len(uv), 4))
    for i in range(0,3):
        coord[i,:] = np.matmul(arm_to_camera,p_model[i,:])
    for i in range(3,7):
        coord[i,:] = np.matmul(rotors_to_camera,p_model[i,:])
    pixelEstimates = np.zeros((len(uv),2))
    for i in range(0,len(uv)):
            pixelEstimates[i,0] = toPixelX(coord[i,0],coord[i,2])*weights[i]
            pixelEstimates[i, 1] = toPixelY(coord[i, 1], coord[i, 2])*weights[i]
    residue = np.linalg.norm(uv-pixelEstimates,axis=1)
    return residue

    # p_model contains estimated 3D-points of marker coordinates


def normal_equations(uv, weights, yaw, pitch,roll):
    #
    # Task 1b: Compute the normal equation terms
    #
    eps = 0.001
    r = residuals(uv,weights,yaw,pitch,roll)
    rDiffYaw = (residuals(uv,weights,yaw+eps,pitch,roll)-r)/eps
    rDiffPitch = (residuals(uv,weights,yaw,pitch+eps,roll)-r)/eps
    rDiffRoll = (residuals(uv,weights,yaw,pitch,roll+eps)-r)/eps

    jacobian = np.array([rDiffYaw.T,rDiffPitch.T,rDiffRoll.T])
    JTJ = jacobian@jacobian.T
    JTr = jacobian@r
    return JTJ, JTr

def gauss_newton(uv, weights, yaw, pitch, roll):
    #
    # Task 1c: Implement the Gauss-Newton method
    #
    max_iter = 28
    alpha = 0.25

    #initial guess
    theta = np.transpose(np.array([yaw,pitch,roll]))
    for iter in range(max_iter):
        JTJ, JTr = normal_equations(uv, weights, yaw, pitch, roll)
        delta = np.linalg.solve(JTJ,-JTr)
        theta =  np.add(theta, alpha*delta)
        yaw = theta[0]
        pitch = theta[1]
        roll = theta[2]
    return yaw, pitch, roll

def levenberg_marquardt(uv, weights, yaw, pitch, roll):
    #
    # Task 2a: Implement the Levenberg-Marquardt method
    JTJ, JTr = normal_equations(uv, weights, yaw, pitch, roll)
    D = np.identity(3)
    la = 1e-3*np.trace(JTJ)/len(JTJ)
    theta = np.transpose(np.array([yaw, pitch, roll]))
    res_old = residuals(uv, weights, theta[0], theta[1], theta[2])
    max_iter = 2
    i = 0
    while i < max_iter:
        JTJ, JTr = normal_equations(uv, weights, yaw, pitch, roll)
        delta = np.linalg.solve(JTJ + la*D,-JTr)
        theta_new = theta + delta
        res_new = residuals(uv,weights,theta_new[0],theta_new[1],theta_new[2])
        if np.linalg.norm(res_old) > np.linalg.norm(res_new):
            theta = theta+delta
            la = la/10
            i +=1
            res_old = res_new
        else:
            la = la*10
        if la > 10e40:
            i += 1
            res_old = res_new
            theta = theta + delta
            la = la / 10
        yaw = theta[0]
        pitch = theta[1]
        roll = theta[2]
    return yaw, pitch, roll

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
def diff_rot_x(radians,eps):
    c_eps = np.cos(radians+eps)
    s_eps = np.sin(radians+eps)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[0, 0, 0, 0],
                     [0, c_eps-c, -s_eps+s, 0],
                     [0, s_eps-s, c_eps-c, 0],
                     [0, 0, 0, 0]])
def diff_rot_y(radians,eps):
    c_eps = np.cos(radians + eps)
    s_eps = np.sin(radians + eps)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c_eps-c, 0, s_eps-s, 0],
                     [0, 0, 0, 0],
                     [-s_eps+s, 0, c_eps-c, 0],
                     [0, 0, 0, 0]])
def diff_rot_z(radians,eps):
    c_eps = np.cos(radians+eps)
    s_eps = np.sin(radians+eps)
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c_eps-c, -s_eps+s, 0, 0],
                     [s_eps-s, c_eps-c, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
def toPixelX(x, z):
    u = 621.01 + 1075.47*x/z
    return u

def toPixelY(x, z):
    v = 362.80 + 1077.22*x/z
    return v