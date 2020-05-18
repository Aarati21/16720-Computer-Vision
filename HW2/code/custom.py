# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:14:38 2019

@author: noron
"""
import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os
import shutil
from sklearn.mixture import GMM
import visual_words

from PIL import Image
def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    NOTE : Please save the dictionary as 'dictionary.npy' in the same dir as the code.
    '''
    data = np.load("../data/train_data.npz")
    alpha = 250
    
    # data is in the form of 2 columns : files and labels. This outputs lists of each
    f_paths = data[data.files[0]]    
    # create temporary directory
    dir = '../Temp_folder/'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
    # Preparing to pass argument lists
    indices = list(range(len(f_paths)))
    alpha_list = [alpha]*len(f_paths)
    file_paths =  ['../data/' + i for i in f_paths]
    arguments = zip(indices, alpha_list, file_paths)
    
    pool = multiprocessing.Pool(num_workers)
    pool.map(visual_words.compute_dictionary_one_image, arguments)
    pool.close()
    pool.join()
 
    # stack all temporary filter responses
    
    temp_list = np.load('../Temp_folder/' + str(0) + '.npy')
    for i in range(1, len(f_paths)):
        temp_file = np.load('../Temp_folder/' + str(i) + '.npy')
        temp_list = np.concatenate((temp_list, temp_file), axis = 0)
            
        
    
    #gaussian mixture models
    gmm = GMM(n_components=75).fit(temp_list)
    d = gmm.means_
    np.save('dictionary.npy', d)
    
    
    
    
def resize_image():
    import Image
# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
imageFile = "zFlowers.jpg"
im1 = Image.open(imageFile)
# adjust width and height to your needs
width = 500
height = 420
# use one of these filter options to resize the image
im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
im5 = im1.resize((width, height), Image.ANTIALIAS) 