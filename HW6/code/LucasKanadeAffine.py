import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    
    p = np.zeros(6)
    
    x = np.arange(It1.shape[0])
    y = np.arange(It1.shape[1])
    
    spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    
    dp = np.array([It1.shape[1]]*6)
    x1, y1, x2, y2 = 0, 0, It.shape[1]-1, It.shape[0]-1
    
    
    while np.sum(dp**2) >= 0.001:
        
        x = np.arange(x1, x2+1)
        y = np.arange(y1, y2+1)
        X, Y = np.meshgrid(x, y)
        Xn = M[0][0]*X + M[0][1]*Y + M[0][2]
        Yn = M[1][0]*X + M[1][1]*Y + M[1][2]
        
        # consider warped images coordinates range
        true_area = ((Xn > 0) & (Xn < It.shape[1]) & (Yn > 0) & (Yn<  It.shape[0]))
              
        I = spline_It1.ev(Yn[true_area], Xn[true_area]).flatten()
        
        It1x = spline_It1.ev(Yn[true_area], Xn[true_area], dx=0, dy=1).flatten()
        It1y = spline_It1.ev(Yn[true_area], Xn[true_area], dx=1, dy=0).flatten()
        
        
        a1 = np.multiply(It1x, X[true_area])
        a2 = np.multiply(It1x, Y[true_area])
        a3 = It1x
        a4 = np.multiply(It1y, X[true_area])
        a5 = np.multiply(It1y, Y[true_area])
        a6 = It1y
        A = np.hstack((a1.reshape(-1, 1), a2.reshape(-1, 1), a3.reshape(-1, 1), a4.reshape(-1, 1), a5.reshape(-1, 1), a6.reshape(-1, 1)))
       
        E = It[true_area].flatten() - I.flatten()
        
        k1 = np.linalg.inv(np.dot(np.transpose(A), A))
        k2 = (A.T).dot(E)
        
        dp = k1.dot(k2)
        
        p += dp.flatten()
        
        M[0][0] = 1.0 + p[0]
        M[0][1] = p[1]
        M[0][2] = p[2]
        M[1][0] = p[3]
        M[1][1] = 1.0 + p[4]
        M[1][2] = p[5]
       
        
    
    return M
