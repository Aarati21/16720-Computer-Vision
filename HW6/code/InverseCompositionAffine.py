import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    x = np.arange(It1.shape[0])
    y = np.arange(It1.shape[1])
    
    spline_It1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It1.shape[1]), It1)
    spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    
    
    dp = np.array([It1.shape[1]]*6)
    x1, y1, x2, y2 = 0, 0, It.shape[1]-1, It.shape[0]-1
    
    x = np.arange(x1, x2+1)
    y = np.arange(y1, y2+1)
    X, Y = np.meshgrid(x, y)
    Itx = spline_It.ev(Y, X, dx=0, dy=1).flatten()
    Ity = spline_It.ev(Y, X,  dx=1, dy=0).flatten()
    X1 = X.flatten()
    Y1 = Y.flatten()
    
    a1 = np.multiply(Itx, X1)
    a2 = np.multiply(Itx, Y1)
    a3 = Itx
    a4 = np.multiply(Ity, X1)
    a5 = np.multiply(Ity, Y1)
    a6 = Ity
    A = np.hstack((a1.reshape(-1, 1), a2.reshape(-1, 1), a3.reshape(-1, 1), a4.reshape(-1, 1), a5.reshape(-1, 1), a6.reshape(-1, 1)))
    while np.sum(dp**2) >= 0.001:
        
       
        Xn = M[0][0]*X + M[0][1]*Y + M[0][2]
        Yn = M[1][0]*X + M[1][1]*Y + M[1][2]
        
        # consider warped images coordinates range
        true_area = ((Xn > 0) & (Xn < It.shape[1]) & (Yn > 0) & (Yn<  It.shape[0]))
              
        I = spline_It1.ev(Yn[true_area], Xn[true_area]).flatten()
        
        
        E = I.flatten() - It[true_area].flatten()
        AN = A[true_area.flatten()]
        
        k1 = np.linalg.inv(np.dot(np.transpose(AN), AN))
        k2 = (AN.T).dot(E)
        
        dp = k1.dot(k2)
        
        M = np.vstack((M, np.array([[0, 0, 1]])))
        dM = np.vstack((np.reshape(dp, (2, 3)), np.array([[0, 0, 1]])))
        
        dM[0, 0] += 1
        dM[1, 1] += 1
        
        M = np.dot(M, np.linalg.inv(dM))
        M = M[:2, :]

    return M