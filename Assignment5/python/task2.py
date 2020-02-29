import numpy as np
import matplotlib.pyplot as plt
from common1 import *
from common2 import *
import scipy.misc
from scipy import ndimage

edge_threshold = 0.013 #todo: choose an appropriate value
blur_sigma     = 3#todo: choose an appropriate value
filename       = '../data/image2_und.jpg'

I_rgb      = plt.imread(filename)
I_rgb      = I_rgb/255.0
I_gray     = rgb2gray(I_rgb)
I_blur     = blur(I_gray, blur_sigma)
Iu, Iv, Im = central_difference(I_blur)
u,v,theta  = extract_edges(Iu, Iv, Im, edge_threshold)
#
# Task 2a: Compute accumulator array H
#
rho = np.zeros((len(u)))
for i in range(0,len(u)):
    rho[i] = u[i]*math.cos(theta[i]) + v[i]*math.sin(theta[i])
bins      = 1000
theta_min = -np.pi
theta_max = np.pi
rho_min = -np.sqrt((350**2 + 800**2))
rho_max = np.sqrt((350**2 + 800**2))
H, xedges, yedges = np.histogram2d(theta,rho, bins=bins, range=[[theta_min, theta_max], [rho_min, rho_max]])
H = H.T
print(yedges)
#convert peaks to corresponding angle and rho
window_size = 140
threshold = 19
peak_theta, peak_rho = return_peak_values(H, window_size, threshold, xedges, yedges)

plt.figure(figsize=[6,8])
plt.subplot(211)
plt.imshow(H, extent=[theta_min, theta_max, rho_min, rho_max], aspect='auto')
plt.xlabel('$\\theta$ (radians)')
plt.ylabel('$\\rho$ (pixels)')
plt.colorbar(label='Votes')
plt.title('Hough transform histogram')
plt.subplot(212)

plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for i in range(len(peak_theta)):
    draw_line(peak_theta[i], peak_rho[i], color='yellow')
#plt.scatter(u, v, s=1, c=theta2,cmap='hsv')
plt.tight_layout()
plt.savefig('out2.png')
plt.show()