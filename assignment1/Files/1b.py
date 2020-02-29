from mpl_toolkits import mplot3d
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
box = np.loadtxt('box.txt')
Pcam = [0,0,6,0]
x = box[:,0]
y = box[:,1]
z = box[:,2]
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(x,y,z,c = 'r',cmap = 'Greens');
#plt.show(block='true')
u = lambda X,Z: 320 + 1000*X/Z
v = lambda Y,Z: 240 + 1100*Y/Z
pix = np.zeros((200,2), dtype=int)
area = np.zeros((200,1))
for i in range(0,200):
	pix[i,0] = u(x[i],Pcam[2]-z[i])
	pix[i,1] = v(y[i],Pcam[2]-z[i])
	area[i] = Pcam[2]-z[i]
	if area[i] > 6:
		area[i] = 2
fig = plt.figure()
print(area)
plt.xlim([0, 640])
plt.ylim([480, 0])
plt.grid()
plt.scatter(pix[:,0],pix[:,1],s = 2*area,c = 'r')
plt.show(block='true')
