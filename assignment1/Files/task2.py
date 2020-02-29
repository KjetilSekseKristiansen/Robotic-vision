import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
import tkinter

def toPixelX(x, z):
    u = 621.01 + 1075.47*x/z
    return u

def toPixelY(x, z):
    v = 362.80 + 1077.22*x/z
    return v

def point(x,y,z):
	return np.array([[x],[y],[z],[1]])
	
def drawPoint(a, **args):
    x = a[0]
    y = a[1]
    z = a[2]

    u = toPixelX(x, z)
    v = toPixelY(y, z)

    plt.plot(u, v, **args)

def drawLine(p0,p1,**args):
	x0 = p0[0]
	y0 = p0[1]
	z0 = p0[2]
	
	x1 = p1[0]
	y1 = p1[1]
	z1 = p1[2]
	
	u0 = toPixelX(x0,z0)
	u1 = toPixelX(x1,z1)
	
	v0 = toPixelY(y0,z0)
	v1 = toPixelY(y1,z1)
	
	plt.plot([u0,u1],[v0,v1],**args)

def translate(x,y,z):
	a = np.identity(4)
	a[0:3,3] = np.transpose([x,y,z])
	return a

def drawCoordinateSystem(T,x,y,z):
	origin = point(0,0,0)
	newOrigin = np.matmul(T,origin)
	
	x_ax = point(x,0,0)
	y_ax = point(0,y,0)
	z_ax = point(0,0,z)
	
	x_trans = np.matmul(T,x_ax)
	y_trans = np.matmul(T,y_ax)
	z_trans = np.matmul(T,z_ax)
	
	drawLine(newOrigin,x_trans,color = 'm')
	drawLine(newOrigin,y_trans,color = 'y')
	drawLine(newOrigin,z_trans,color = 'g')
	
def rotateX(theta):
	ang = theta*np.pi/180
	c = np.cos(ang)
	s = np.sin(ang)
	
	a = np.identity(4)
	rot = np.array([[c, -s],[s,c]])
	a[1:3,1:3] = rot
	return a
	
def rotateY(theta):
	ang = theta*np.pi/180
	c = np.cos(ang)
	s = np.sin(ang)
	a = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
	return a

def rotateZ(theta):
	ang = theta*np.pi/180
	c = np.cos(ang)
	s = np.sin(ang)
	a = np.identity(4)
	rot = np.array([[c, -s],[s,c]])
	a[0:2,0:2] = rot
	return a

platformToBase = np.matmul(translate(0.05725,0.05725,0),rotateZ(11))
baseToHinge = np.matmul(translate(0,0,0.325),rotateY(28.87))
hingeToArm = translate(0,0,-0.0552)
armToRotors = np.matmul(translate(0.653,0,-0.0312),rotateX(-0.5))
camToObj = np.loadtxt('heli_pose.txt')
heliPoints = np.loadtxt('heli_points.txt')
img = plt.imread('quanser.jpg')
imgplot = plt.imshow(img)


screw1 = np.matmul(camToObj,point(0,0,0))
screw2 = np.matmul(camToObj,point(0,0.1145,0))
screw3 = np.matmul(camToObj,point(0.1145,0,0))
screw4 = np.matmul(camToObj,point(0.1145,0.1145,0))

drawPoint(screw1,markersize ='12' ,marker = '.',color = 'r')
drawPoint(screw2,markersize ='12' ,marker = '.',color = 'r')
drawPoint(screw3,markersize ='12' ,marker = '.',color = 'r')
drawPoint(screw4,markersize ='12' ,marker = '.',color = 'r')
drawCoordinateSystem(camToObj,0.1145,0.1145,0.1145)

camToBase = np.matmul(camToObj,platformToBase)
drawCoordinateSystem(camToBase,0.1145,0.1145,0.1145)

camToHinge = np.matmul(camToBase,baseToHinge)
drawCoordinateSystem(camToHinge,0.1145,0.1145,0.1145)

camToArm = np.matmul(camToHinge,hingeToArm)
arm1 = np.matmul(camToArm,np.transpose(heliPoints[0,:]))
arm2 = np.matmul(camToArm,np.transpose(heliPoints[1,:]))
arm3 = np.matmul(camToArm,np.transpose(heliPoints[2,:]))


drawPoint(np.transpose(arm1),markersize ='12' ,marker = '.',color = 'r')
drawPoint(np.transpose(arm2),markersize ='12' ,marker = '.',color = 'r')
drawPoint(np.transpose(arm3),markersize ='12' ,marker = '.',color = 'r')
drawCoordinateSystem(camToArm,0.1145,0.1145,0.1145)

camToRotors = np.matmul(camToArm,armToRotors)
rotor1 = np.matmul(camToRotors,np.transpose(heliPoints[3,:]))
rotor2 = np.matmul(camToRotors,np.transpose(heliPoints[4,:]))
rotor3 = np.matmul(camToRotors,np.transpose(heliPoints[5,:]))
rotor4 = np.matmul(camToRotors,np.transpose(heliPoints[6,:]))

drawPoint(np.transpose(rotor1),markersize ='12' ,marker = '.',color = 'r')
drawPoint(np.transpose(rotor2),markersize ='12' ,marker = '.',color = 'r')
drawPoint(np.transpose(rotor3),markersize ='12' ,marker = '.',color = 'r')
drawPoint(np.transpose(rotor4),markersize ='12' ,marker = '.',color = 'r')
drawCoordinateSystem(camToRotors,0.1145,0.1145,0.1145)

plt.show()
print(heliPoints)
print(np.transpose(heliPoints[1,:]))
print(screw1)
print(arm1)