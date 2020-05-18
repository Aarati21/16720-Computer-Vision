import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    
    A = np.zeros((2*p1.shape[1], 9))
    for i in range(p1.shape[1]):
        A[2*i+1] = [0, 0, 0, -p2[0, i], -p2[1, i], -1, p1[1, i]*p2[0, i], p1[1, i]*p2[1, i], p1[1, i]]
        A[2*i] = [p2[0, i], p2[1, i], 1, 0, 0, 0, -p1[0, i]*p2[0, i], -p1[0, i]*p2[1, i], -p1[0, i]]

       
    _, _, V = np.linalg.svd(A)
    H2to1 = V[-1,:]/ V[-1,-1]
    H2to1 = H2to1.reshape(3,3)
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
   
    
     
    
    p1 = np.empty((2, 4))
    p2 = np.empty((2, 4))
    max_inliers = 0
    
    for i in range(num_iter):
        rand_ind = np.random.choice(len(matches), 4, replace = False)
        
        for j in range(len(rand_ind)):
            p1[:, j] = locs1[matches[rand_ind[j], 0], :2]
            p2[:, j] = locs2[matches[rand_ind[j], 1], :2]
            
        H = computeH(p1, p2)
        X1 = np.vstack((locs1[matches[:,0],:2].T, np.ones((1, len(matches)))))
        U1 = np.vstack((locs2[matches[:,1],:2].T, np.ones((1, len(matches)))))
        pp1 = np.matmul(H, U1)
        ssd = np.sum(np.square(X1 - (pp1/pp1[2,:])), axis=0)
        inliers = 0
        for i in ssd:
            if i<tol**2:
                inliers +=1
        if inliers>max_inliers:
            max_inliers = inliers
            bestH = H
                    
    print(max_inliers)
    return bestH
       


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png') 
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

