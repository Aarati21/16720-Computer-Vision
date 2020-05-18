'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import submission
import helper

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

tc = np.load('../data/templeCoords.npz')
data = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

pts1 = data['pts1']
pts2 = data['pts2']

x1 = tc['x1']
y1 = tc['y1']



K1 = intrinsics['K1']
K2 = intrinsics['K2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = max(im1.shape)

F = submission.eightpoint(pts1, pts2, M)
E = submission.essentialMatrix(F, K1, K2)
X = np.zeros((len(x1), 1))
Y = np.zeros((len(x1), 1))

for i in range(len(x1)):
    X[i], Y[i] = submission.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
p1 = np.hstack((x1, y1))
p2 = np.hstack((X, Y))

M1 = np.array([[1.0, 0, 0, 0], 
               [0, 1.0, 0, 0], 
               [0, 0, 1.0, 0]])


M2s = helper.camera2(E)

C1 = K1.dot(M1)

min_error = np.inf
M_final = np.array([])
P_final = 0
C_final = 0
for i in range(M2s.shape[2]):
    C2 = K2.dot(M2s[:,:,i])
    P, error = submission.triangulate(C1, p1, C2, p2)        
    if error < min_error:
        if (P[:,-1]>=0).all():
            min_error = error
            M_final = M2s[:,:,i]
            P_final = P
            C_final = C2
np.savez('../results/q4_2.npz', F=F, M1 =M1, M2 = M_final, C1 =C1, C2 = C_final)  
     
print('Error', min_error)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-0.8, 0.8)
ax.set_ylim3d(-0.8, 0.8)
ax.set_zlim3d(np.min(P_final[:, 2]), np.max(P_final[:, 2]))
ax.scatter(P_final[:, 0], P_final[:, 1], P_final[:, 2], c='b', marker='o', s=40)
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
