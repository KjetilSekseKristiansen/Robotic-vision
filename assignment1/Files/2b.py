from mpl_toolkits import mplot3d
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt

heliPose = np.loadtxt('heli_pose.txt')
points = np.array([[0, 0, 0, 1],[0.1145, 0, 0, 1],[0.1145, 0.1145,0,1],[0,0.1145,0,1]])
pix = np.zeros((len(points),2),dtype = int)
area = np.zeros((len(points),1))

coord = np.array([[1,0,0,1],[0,-1,0,1],[0,0,1,1]])
newcoord = zeros((3,4),dtype = float)
coordPix = np.zeros((3,2),dtype=int)

u = lambda X,Z: 621.01 + 1075.47*X/Z
v = lambda Y,Z: 362.8 + 1077.22*Y/Z
for i in range(0,len(points)):
	points[i] = np.transpose(np.matmul(heliPose,np.transpose(points[i])))
	pix[i,0] = u(points[i,0],points[i,2])
	pix[i,1] = v(points[i,1],points[i,2])
	area[i] = points[i,1]
for i in range(0,3):
	newcoord[i] = np.transpose(np.matmul(heliPose,np.transpose(coord[i])))
	print('displaying transformed coordinate')
	print(newcoord[i])
	coordPix[i,0] = u(newcoord[i,0],newcoord[i,2])
	coordPix[i,1] = v(newcoord[i,1],newcoord[i,2])
	#normalize pixel coordinates
	EucDist = np.sqrt((coordPix[i,0]-269)**2 + (coordPix[i,1]-521)**2)
	coordPix[i,0] = 150*coordPix[i,0]/EucDist + 269
	coordPix[i,1] = 150*coordPix[i,1]/EucDist + 521
	print(EucDist)
	print(coordPix[i])
axes = np.array(['x','y','z'])
coordPix[0,0] = 421
coordPix[0,1] = 498

img = plt.imread('quanser.jpg')
imgplot = plt.imshow(img)
for i in range(0,3):
	plt.plot([269,coordPix[i,0]],[521,coordPix[i,1]],label = axes[i])
	print("plotted one line")
print(pix)
plt.xlim([100,600])
plt.ylim([600,300])
plt.grid()
plt.legend()
plt.scatter(pix[:,0],pix[:,1],s = 30*area,c = 'r')
plt.show(block='true')