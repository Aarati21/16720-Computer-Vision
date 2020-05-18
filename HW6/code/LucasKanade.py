import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	
 
    p = p0
    
    
    spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    dp = np.array([It.shape[1], It.shape[0]])
    
    while np.sum(dp**2) >= 0.001:
        x = np.arange(rect[0], rect[2]+1)
        y = np.arange(rect[1], rect[3]+1)
        X, Y = np.meshgrid(x, y)
        It_warp = spline_It.ev(Y, X)
        
        Xp = np.add(X.reshape(-1,1), p[0])
        Yp = np.add(Y.reshape(-1,1), p[1])
       
        It1_warp = spline_It1.ev(Yp, Xp)
        
        E = It_warp.flatten() - It1_warp.flatten()   
        
        It1x = spline_It1.ev(Yp, Xp, dx=0, dy=1).flatten()
        It1y = spline_It1.ev(Yp, Xp, dx=1, dy=0).flatten()
        
        A = np.hstack((It1x.reshape(-1,1), It1y.reshape(-1,1)))
        
        
        a1 = np.linalg.inv(np.dot(np.transpose(A), A))
        a2 = (A.T).dot(E)
        
        dp = a1.dot(a2)
        
        p += dp.flatten()
        
        
    return p

        

        
   
