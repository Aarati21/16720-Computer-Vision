import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    
    
    DoG_levels = levels[1:]
    
    DoG_pyramid = (gaussian_pyramid[:,:,1]-gaussian_pyramid[:,:,0])[:,:,np.newaxis]
    for i in range(1,len(DoG_levels)):
        DoG_layer = (gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
        DoG_pyramid = np.concatenate((DoG_pyramid, DoG_layer[:,:, np.newaxis]), axis=2)
    
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    dxx = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 2,0, ksize=5)
    dyy = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 0,2, ksize=5)
    dxy = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 1,1, ksize=5)
    
    principal_curvature = (dxx + dyy)**2 / ((dxx*dyy) - (dxy)**2)
    
    return principal_curvature


   
    return locsDoG.astype(int)
def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    h, w, l = DoG_pyramid.shape
    locsDoG=[]
    #split pyramid along the layer axis
    DP = np.split(DoG_pyramid, DoG_pyramid.shape[2], axis=2)
    DP = [i.reshape(h,w) for i in DP]
    PC = np.split(principal_curvature, principal_curvature.shape[2], axis=2)
    PC = [i.reshape(h,w) for i in PC]
    
    
   
    for i in range(len(DP)):
        n_cells = []
        
        #compute neighbors in same level
        n_cells = [
                    DP[i][1:-1,:-2], DP[i][:-2,:-2], DP[i][:-2,1:-1], DP[i][:-2,2:],
                    DP[i][1:-1,2:], DP[i][2:,2:], DP[i][2:,1:-1], DP[i][2:,:-2]]
        #compute neighbors below and above
        if i>=1:
            n_cells.append(DP[i-1][1:-1, 1:-1])
        if i<=l-2:
            n_cells.append(DP[i+1][1:-1, 1:-1])
        n_cells = np.asarray(n_cells)    
        
        #satisfy conditions

        valid_cond = ((np.abs(DP[i][1:-1, 1:-1])> th_contrast) & (np.abs(PC[i][1:-1, 1:-1]) <= th_r) & (((DP[i][1:-1, 1:-1] > np.max(n_cells, axis=0)) | (DP[i][1:-1, 1:-1] < np.min(n_cells, axis=0)))))
        valid_points = np.where(valid_cond == True)
        for j in range(len(valid_points[0])):
            locsDoG.append((valid_points[1][j]+1, valid_points[0][j]+1, DoG_levels[i]))
    return locsDoG     
    


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    
    pc = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, pc, th_contrast, th_r)
   
    return locsDoG, gauss_pyramid






if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
   
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
   
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i, j, k in locsDoG:
        cv2.circle(im, (i, j), 1, [0, 255, 0])
    
    im = cv2.resize(im, (im.shape[1], im.shape[0]))
    cv2.imshow("keypoint", im)
    cv2.waitKey(0)