import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage.morphology as morph
from InverseCompositionAffine import InverseCompositionAffine
from LucasKanadeAffine import LucasKanadeAffine
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2) 
    
    if len(M)<3:
        M = np.vstack((M, np.array([0, 0, 1])))
    M = np.linalg.inv(M)
    
    x1, y1, x2, y2 = 0, 0, image2.shape[1]-1, image2.shape[0]-1
    
    spline_It = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    spline_It1 = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)
    
    x = np.arange(x1, x2+1)
    y = np.arange(y1, y2+1)
    X, Y = np.meshgrid(x, y)
    Xn = M[0, 0]*X + M[0,1]*Y + M[0,2]
    Yn = M[1, 0]*X + M[1,1]*Y + M[1,2]
    
    
    It1= spline_It.ev(Yn, Xn)
    It2 = spline_It1.ev(Y, X)
    true_area = ((Xn > 0) & (Xn < image1.shape[1]) & (Yn > 0) & (Yn <  image1.shape[0]))
    It1[~true_area]  = 0   
    It2[~true_area]  = 0  
    mask[(abs(It2-It1) > 0.15) & (It2 !=0)] = 1

    mask = morph.binary_dilation(mask).astype(mask.dtype)
    
    
    return mask





