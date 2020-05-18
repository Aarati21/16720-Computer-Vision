import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import math
import shutil
from sklearn.mixture import GMM
def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # convert into floating point with range[0,1]
    if image.dtype!='float' or np.amax(image)>1:
        image = image.astype('float')/255
        
    # convert gray scale images into 3 channel images   
    if len(image.shape)!=3:
        image = np.repeat(image[:,:, np.newaxis], 3, axis = 2)
        
    # check if number of channels is 3. If not just take the slice upto 3
    if image.shape[2]!=3:
        image = image[:,:,:3]
        
    # convert image into the lab color space
    image = skimage.color.rgb2lab(image) 
    
    f_images = np.empty((image.shape[0], image.shape[1], 0))
    scales = [1, 2, 4, 8, 8*math.sqrt(2)]
    axes = [0, 1, 2]
    for i in scales:
        for j in axes:
        # Gaussian filter
            f_images = np.concatenate((f_images, scipy.ndimage.filters.gaussian_filter(image[:,:,j], i)[:, :, np.newaxis]), axis=2)
        for j in axes:
        # Laplacian filter
            f_images = np.concatenate((f_images, scipy.ndimage.filters.gaussian_laplace(image[:,:,j], i)[:, :, np.newaxis]), axis=2)
        # Gaussian filter in x direction
        for j in axes:
            f_images = np.concatenate((f_images, scipy.ndimage.filters.gaussian_filter(image[:,:,j], i, [0, 1])[:, :, np.newaxis]), axis=2)
        # Gaussian filter in y direction  
        for j in axes:
            f_images = np.concatenate((f_images, scipy.ndimage.filters.gaussian_filter(image[:,:,j], i, [1, 0])[:, :, np.newaxis]), axis=2)
    return f_images


def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    f_image = extract_filter_responses(image)
    h = f_image.shape[0]
    w = f_image.shape[1]
    pixels = h*w
    f_image = f_image.reshape(pixels, -1)
    
    #computes distance with every point registered in the dictionary and outputs it in a row for each pixel
    compute_distance = scipy.spatial.distance.cdist(f_image, dictionary, metric ='cosine')
    wordmap = np.argmin(compute_distance, axis = 1)
    wordmap = wordmap.reshape(h, w)
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    #extract filter responses
    i,alpha,image_path = args
    image = imageio.imread(image_path)
    f_image = extract_filter_responses(image)
    
    #convert image into dimensions in terms of pixels
    h = f_image.shape[0]
    w = f_image.shape[1]
    f_image = f_image.reshape(h*w, -1)
    k = np.random.choice((h*w), alpha, replace = True)
    sampled_response = f_image[k, :]
    
    
    #save filter response to patches
    np.save('../Temp_folder1/' + str(i) + '.npy', sampled_response)
    

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
    clusters = 165
    # data is in the form of 2 columns : files and labels. This outputs lists of each
    f_paths = data[data.files[0]]    
    # create temporary directory
    dir = '../Temp_folder1/'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
    # Preparing to pass argument lists
    indices = list(range(len(f_paths)))
    alpha_list = [alpha]*len(f_paths)
    file_paths =  ['../data/' + i for i in f_paths]
    arguments = zip(indices, alpha_list, file_paths)
   
        
    pool = multiprocessing.Pool(num_workers)
    pool.map(compute_dictionary_one_image, arguments)
    pool.close()
    pool.join()
    
    # stack all temporary filter responses
    
    temp_list = np.load('../Temp_folder1/' + str(0) + '.npy')
    
    for i in range(1, len(f_paths)):
        temp_file = np.load('../Temp_folder1/' + str(i) + '.npy')
        temp_list = np.concatenate((temp_list, temp_file), axis = 0)
            
        
    # kmeans
    k = sklearn.cluster.KMeans(n_clusters = clusters, n_jobs = num_workers).fit(temp_list)
    d = k.cluster_centers_
    np.save('dictionary.npy', d)
    
    