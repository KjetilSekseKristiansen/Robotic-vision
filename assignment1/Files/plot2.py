import numpy as np
import matplotlib
import tkinter
import matplotlib.pyplot as plt


def toPixelX(x, z):
    u = 621.01 + 1075.47*x/z
    return u

def toPixelY(x, z):
    v = 362.80 + 1077.22*x/z
    return v

def draw_line_in_room(p1, p2, **args):
    x = p1[0,0]
    y = p1[1,0]
    z = p1[2,0]


    x1 = p2[0,0]
    y1 = p2[1,0]
    z1 = p2[2,0]

    u1 = toPixelX(x, z)
    v1 = toPixelY(y, z)
    u2 = toPixelX(x1, z1)
    v2 = toPixelY(y1, z1)
    plt.plot([u1, u2],[v1, v2], **args)

def point(x, y, z):
    return np.array([[x],[y],[z],[1]])

def drawPoint(a, **args):
    x = a[0,0]
    y = a[1,0]
    z = a[2,0]

    u = toPixelX(x, z)
    v = toPixelY(y, z)

    plt.plot(u, v, **args)

def drawAlongAxis(T, x, y, z):
    origin = point(0,0,0)
    trans_origin = T@origin


    x_ax = point(x, 0,   0)
    y_ax = point(0,  y,  0)
    z_ax = point(0,   0,  z)


    trans_x = T@x_ax
    trans_y = T@y_ax
    trans_z = T@z_ax


    draw_line_in_room(trans_origin, trans_x, color = 'b')
    draw_line_in_room(trans_origin, trans_y, color = 'r')
    draw_line_in_room(trans_origin, trans_z, color = 'g')

def translate(x, y, z):
    trans = np.array([[1, 0, 0, x],
                       [0, 1, 0, y],
                       [0, 0, 1, z],
                       [0, 0, 0, 1]])
    return trans

def rotateX(theta_d):
    theta = theta_d*np.pi/180
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[1, 0, 0, 0],
                    [0, c, -s, 0],
                    [0, s, c, 0],
                    [0, 0, 0, 1]])
    return rot


def rotateY(theta_d):
    theta = theta_d*np.pi/180
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, 0, s, 0],
                    [0, 1, 0, 0],
                    [-s, 0, c, 0],
                    [0, 0, 0, 1]])
    return rot

def rotateZ(theta_d):
    theta = theta_d*np.pi/180
    c = np.cos(theta)
    s = np.sin(theta)
    rot = np.array([[c, -s, 0, 0],
                    [s, c, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return rot


camToObj = np.loadtxt('heli_pose.txt')




## 2b
plt.figure('task 2b')
img = plt.imread('quanser.jpg')
imgplot = plt.imshow(img)
plt.axis('scaled')
plt.xlim([100,600])
plt.ylim([600, 300])
plt.xticks([100, 200, 300, 400, 500, 600])
plt.yticks([600, 550, 500, 450, 400, 350, 300])
plt.tight_layout()


screw1 = camToObj@point(0,0,0)
screw2 = camToObj@point(0.1145, 0, 0)
screw3 = camToObj@point(0.1145, 0.1145, 0)
screw4 = camToObj@point(0, 0.1145, 0)

drawPoint(screw1, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw2, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw3, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw4, markersize ='12' , marker = '.',color = 'r')


drawAlongAxis(camToObj, 0.1145, 0.1145, 0.1145)


## 2c

psi = 11.77
platformToBase = translate(0.1145/2, 0.1145/2, 0)@rotateZ(psi)
cameraToBase = camToObj@platformToBase

plt.figure('task 2d')
imgplot = plt.imshow(img)
plt.axis('scaled')
plt.xlim([100,600])
plt.ylim([600, 300])
plt.xticks([100, 200, 300, 400, 500, 600])
plt.yticks([600, 550, 500, 450, 400, 350, 300])
plt.tight_layout()

drawPoint(screw1, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw2, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw3, markersize ='12' , marker = '.',color = 'r')
drawPoint(screw4, markersize ='12' , marker = '.',color = 'r')

drawAlongAxis(cameraToBase, 0.1, 0.1, 0.1)


##2f

plt.figure('Task 2f')
imgplot  = plt.imshow(img)
plt.axis('scaled')
plt.xlim([0, 1280])
plt.ylim([720, 0])
plt.xticks([0, 320, 640, 960, 1280])
plt.yticks([720, 540, 360, 180, 0])

theta = 28.87
phi = -0.85
points = np.loadtxt('heli_points.txt')
print(points)

#move from camera to platform
drawAlongAxis(camToObj, 0.1, 0.1, 0.1)

#move from platform to base
drawAlongAxis(cameraToBase, 0.1, 0.1, 0.1)

#Move from base to hinge
baseToHinge = translate(0, 0, 0.325)@rotateY(theta)
cameraToHinge = cameraToBase@baseToHinge

drawAlongAxis(cameraToHinge, 0.1, 0.1, 0.1)

#move from hinge to arm

hingeToArm = translate(0, 0, -0.0552)
cameraToArm = cameraToHinge@hingeToArm

for i in range(0, 3):
    toBeDrawn = point(points[i, 0], points[i, 1], points[i, 2])
    drawPoint(cameraToArm@toBeDrawn, markersize ='12' , marker = '.',color = 'Red')


drawAlongAxis(cameraToArm, 0.1, 0.1, 0.1)

#move from arm to rotors

armToRotors = translate(0.653, 0, -0.0312)@rotateX(phi)
cameraToRotors = cameraToArm@armToRotors

for i in range(3, 7):
    toBeDrawn = point(points[i, 0], points[i, 1], points[i, 2])
    drawPoint(cameraToRotors@toBeDrawn, markersize ='12' , marker = '.',color = 'Red')

drawAlongAxis(cameraToRotors, 0.1, 0.1, 0.1)








######################
plt.show()