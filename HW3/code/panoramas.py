import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    np.save('../results/q6_1.npy', H2to1)
    
    #Warps img2 into img1’s reference frame using the aforementioned perspective warping function
    out_size = (im1.shape[1], im1.shape[0])
    warp_im2 = cv2.warpPerspective(im2, H2to1, out_size)
    cv2.imwrite('../results/6.1.jpg', warp_im2)
    cv2.imshow('6.1', warp_im2)
      
    blended_image = np.maximum(im1, warp_im2)
    cv2.imwrite('../results/panorama6.1.jpg',blended_image)
    cv2.imshow('6.1p',blended_image)
    
        
    return blended_image


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    output_width = 1700#arbitary
    h = im2.shape[1]
    w = im2.shape[0]
    corners = [0, 0, 1, h-1, 0, 1, h-1, w-1, 1, 0, w-1, 1]
    corners = np.asarray(corners)
    corners = corners.reshape(4,3)
    warped_final = np.empty((4,2))
    warped_corners = np.dot(H2to1, corners.T)
    warped_final[:,0] = warped_corners[0,:]/warped_corners[2,:]
    warped_final[:,1] = warped_corners[1,:]/warped_corners[2,:]
    scale = (np.max(warped_final[:,0]))/output_width
    output_height = int(((np.max(warped_final[:,1]))-(np.min(warped_final[:,1])))/scale)+1
    outsize = (output_width,output_height)
    M=np.array([[1,0,0],[0,1,-(np.min(warped_final[:,1]))],[0,0,scale]],dtype=float)
    warpim1 = cv2.warpPerspective(im1, M, outsize) 
    warpim2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), outsize)
    pano_im = np.maximum(warpim1, warpim2)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    cv2.imshow('6.2', pano_im)
    return pano_im
    


    

def generatePanorama(im1, im2):
    '''
    Returns a panorama of im1 and im2 without cliping.
    ''' 
    ######################################
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    blended_image = imageStitching(im1, im2, bestH)
    panorama = imageStitching_noClip(im1, im2, bestH)
    cv2.imwrite('../results/q6_3.jpg',panorama)
    cv2.imshow('6.3p',panorama)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    generatePanorama(im1, im2)
    
