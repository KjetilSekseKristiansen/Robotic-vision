from typing import Any, Union

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def spacialX(u,cx,fx):
    return (u-cx)/fx
def spacialY(v,cy,fy):
    return (v-cy)/fy
def xDistortion(k1,k2,k3,p1,p2,x,y):
    r_squared =x**2+y**2
    delta_x = (k1*r_squared + k2*r_squared**2 + k3*r_squared**3)*x + p2*(r_squared+2*x**2) + 2*p1*x*y
    return delta_x
def yDistortion(k1,k2,k3,p1,p2,x,y):
    r_squared = x**2 + y**2
    delta_y = (k1*r_squared + k2*r_squared**2 + k3*r_squared**3)*y + p1*(r_squared+2*y**2) + 2*p2*x*y
    return delta_y
def projectUndistortedX(cx,fx,x,delta_x):
    return cx + fx*(x+delta_x)
def projectUndistortedY(cy,fy,y,delta_y):
    return cy + fy*(y+delta_y)
def undistortX(cx,fx,k1,k2,k3,p1,p2,u,v):
    x = spacialX(u , cx , fx)
    y = spacialY(v , cy , fy)
    delta_x = xDistortion(k1,k2,k3,p1,p2,x,y)
    return int(np.round(projectUndistortedX(cx,fx,x,delta_x)))
def undistortY(cx,fx,k1,k2,k3,p1,p2,u,v):
    x = spacialX(u, cx, fx)
    y = spacialY(v , cy , fy)
    delta_y = yDistortion(k1,k2,k3,p1,p2,x,y)
    return int(np.round(projectUndistortedY(cx,fx,y,delta_y)))

#camera intrinsics and extrinsics
fx = 9.842439e+02
cx = 6.900000e+02
fy = 9.808141e+02
cy = 2.331966e+02
k1 = -3.728755e-01
k2 = 2.037299e-01
p1 = 2.219027e-03
p2 = 1.383707e-03
k3 = -7.233722e-02
img  = mpimg.imread('data/kitti.jpg')
y,x,_ = np.shape(img)
newImg = np.zeros((y,x,3),dtype = 'int')
for i in range(0,y):
    for j in range(0,x):
        newX = undistortX(cx,fx,k1,k2,k3,p1,p2,j,i)
        newY = undistortY(cy, fy, k1, k2, k3, p1, p2, j, i)
        if(newY > y):
            print(newY)
        newImg[i,j,:] =img[newY,newX,:]
plt.subplot(211)
plt.imshow(newImg)
plt.suptitle('undistorted image top and original bottom')
plt.subplot(212)
plt.imshow(img)
plt.show()