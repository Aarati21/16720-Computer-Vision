'''
Q1
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


#2.1
F1 = submission.eightpoint(pts1, pts2, M)
#helper.displayEpipolarF(im1, im2, F1)

#2.2

'''
error = 1000
ri = 0
FF = np.array([])

for j in range(10000):
    
    rand_index = np.random.choice(len(pts1), 7, replace = False)
    p1 = np.zeros((7,2))
    p2 = np.zeros((7,2))
    for i in range(len(rand_index)):
    
        p1[i] = pts1[rand_index[i]]
        p2[i] = pts2[rand_index[i]]

    F2 = submission.sevenpoint(p1, p2, M)
    
    
    for i in F2:
        
        #helper.displayEpipolarF(im1, im2, i)
        MSE = np.square(np.subtract(i,F1)).mean() 
        
        if MSE<error:
            error = MSE
            FF = i
            ri = rand_index
print(error)            
print(ri) 
print(p1, p2)
print(FF)

'''

rand_index = [77, 44, 60, 39, 26, 42, 61]
p1 = np.zeros((7,2))
p2 = np.zeros((7,2))
for i in range(len(rand_index)):
    
    p1[i] = pts1[rand_index[i]]
    p2[i] = pts2[rand_index[i]]

F2 = submission.sevenpoint(p1, p2, M)
    
helper.displayEpipolarF(im1, im2, F2[2])
 

'''
E = submission.essentialMatrix(F, K1, K2)
np.savez('../results/q4_1.npz', F = F, pts1 = pts1, pts2 =pts2)
helper.epipolarMatchGUI(im1,im2,F)

'''
