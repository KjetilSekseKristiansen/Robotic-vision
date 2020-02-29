from mpl_toolkits import mplot3d
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
box = np.loadtxt('box.txt')
pix = np.zeros((200,2), dtype=int)
area = np.zeros((200,1))
coord = np.array([[1,0,0,1],[0,1,0,1],[0,0,-1,1]])
newcoord = zeros((3,4),dtype = float)
print(coord[2])
coordPix = np.zeros((3,2),dtype=int)
#define transformation matrices
Tz = np.identity(4)
Tz[2,3] = 5
theta = -pi/6
RotZ = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
Rz = np.identity(4)
Rz[0:2,0:2] = RotZ
RotY = np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
Ry = np.identity(4)
Ry[0:3,0:3] = RotY
Rx = np.identity(4)
Rx[1:3,1:3] = RotZ
#convert from camera to pinhole	
u = lambda X,Z: 320 + 1000*X/Z
v = lambda Y,Z: 240 + 1100*Y/Z
for i in range(0,200):
	box[i] = np.transpose(np.matmul(Ry,np.transpose(box[i])))
	box[i] = np.transpose(np.matmul(Rx,np.transpose(box[i])))
	pix[i,0] = u(box[i,0],6-box[i,2])
	pix[i,1] = v(box[i,1],6-box[i,2])
	area[i] = 6-box[i,2]
	if area[i] > 6:
		area[i] = 2
		
for i in range(0,3):
	newcoord[i] = np.transpose(np.matmul(Ry,np.transpose(coord[i])))
	newcoord[i] = np.transpose(np.matmul(Rx,np.transpose(newcoord[i])))
	print(newcoord[i])
	coordPix[i,0] = u(newcoord[i,0],6-newcoord[i,2])
	coordPix[i,1] = v(newcoord[i,1],6-newcoord[i,2])
fig = plt.figure()
plt.grid()
plt.scatter(pix[:,0],pix[:,1],s = 2*area,c = 'r')
print(coordPix)
axes = np.array(['x','y','z'])
for i in range(0,3):
	plt.plot([320,coordPix[i,0]],[240,coordPix[i,1]],label = axes[i])
	print("plotted one line")
plt.xlim([0, 640])
plt.ylim([480, 0])
plt.legend()
plt.show(block='true')