# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:58:59 2019

@author: noron
"""

import numpy as np
import cv2
import BRIEF
import matplotlib.pyplot as plt






if __name__ == '__main__':
    
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = BRIEF.briefLite(im1)
    angles = range(0, 361, 10)
    match_total = []
    for i in angles:
        
        A = cv2.getRotationMatrix2D((im1.shape[1]/2, im1.shape[0]/2), i, 1)
        im2 = cv2.warpAffine(im1, A, (im1.shape[1], im1.shape[0]))
        locs2, desc2 = BRIEF.briefLite(im2)
        matches = BRIEF.briefMatch(desc1, desc2)
        match_total.append(len(matches))
    
    plt.figure()
    plt.bar(angles, match_total, align='center', alpha=0.5, color='k')
    plt.ylabel('Matches')
    plt.xlabel('Image rotation in degrees')
    plt.title('Rotation vs Number_of_correct_Matches')
    plt.show()
    


