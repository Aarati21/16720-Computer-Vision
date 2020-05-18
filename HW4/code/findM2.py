'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


import numpy as np
import submission
import helper
import cv2

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

data_corr = np.load('../data/some_corresp.npz')
pts1 = data_corr['pts1']
pts2 = data_corr['pts2'] 

data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']

image_dim = [im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]]
M = max(image_dim)

F = submission.eightpoint(pts1, pts2, M)
E = submission.essentialMatrix(F, K1, K2)

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
    P, error = submission.triangulate(C1, pts1, C2, pts2)
    
        
    if error < min_error:
        if (P[:,-1]>=0).all():
            min_error = error
            M_final = M2s[:,:,i]
            P_final = P
            C_final = C2
np.savez('../results/q3_3.npz', M2 = M_final, C2 = C_final, P = P_final)       
print('Error', min_error)



