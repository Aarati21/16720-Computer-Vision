import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing

def build_recognition_system(num_workers=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''



    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    f_paths = train_data[train_data.files[0]]  
    labels = train_data[train_data.files[1]]  

    layer_num = 3
    
    
    f =  ['../data/' + i for i in f_paths]
    d = [dictionary]*len(f)
    l = [layer_num]*len(f)
    k = [dictionary.shape[0]]*len(f)
    args = zip(f, d, l, k)
    pool = multiprocessing.Pool(num_workers)
    features_list = pool.map(get_image_feature1, args)
    pool.close()
    pool.join()
    np.savez("trained_system_features.npz", features=features_list)

    
   
    
    
    features = np.asarray(features_list[0])
    for i in range(1, len(features_list)):
        features = np.vstack((features, features_list[i]))
             
    np.savez("trained_system.npz", features=features, labels=labels, dictionary=dictionary, SPM_layer_num=layer_num)
    
       
    
def evaluate_recognition_system(num_workers=4):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    test_data = np.load("../data/test_data.npz")
    test_path = test_data[test_data.files[0]]
    test_labels = test_data[test_data.files[1]]
    
    trained_system = np.load("trained_system.npz")
    trained_features = trained_system[trained_system.files[0]]
    trained_labels = trained_system[trained_system.files[1]]
    dictionary = trained_system[trained_system.files[2]]
    layer = trained_system[trained_system.files[3]]
    
    f =  ['../data/' + i for i in test_path]
    d = [dictionary]*len(f)
    l = [layer]*len(f)
    k = [dictionary.shape[0]]*len(f)
    args = zip(f, d, l, k)
    pool = multiprocessing.Pool(num_workers)
    features_test_list = pool.map(get_image_feature1, args)
    pool.close()
    pool.join()
    
    


    
    features = np.asarray(features_test_list[0])
    for i in range(1, len(features_test_list)):
       features = np.vstack((features, features_test_list[i]))
    
    np.savez("t_system.npz", features=features)

    
    
   
    #test_data = np.load("../data/test_data.npz")
    #test_labels = test_data[test_data.files[1]]
    conf = np.zeros((8,8))
    index = 0
    
   
    #trained_system = np.load("trained_system.npz")
    #trained_features = trained_system[trained_system.files[0]]
    
    #trained_labels = trained_system[trained_system.files[1]]
    for i in features:
        pred_label = trained_labels[np.argmax(distance_to_set(i, trained_features))]
        conf[test_labels[index], pred_label]+=1
        index+=1
    
    correctly_classified = sum(conf[i][i] for i in range(len(conf)))
    accuracy = correctly_classified/np.sum(conf)
    return conf, accuracy
    
def get_image_feature1(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)/3))
    '''
    file_path,dictionary,layer_num,K = args
    image = imageio.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    return feature


def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)/3))
    '''
    
    image = imageio.imread(file_path)
    image = image.astype('float')/255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap,layer_num,K)
    return feature

def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    
    
    
    return np.sum(np.minimum(word_hist, histograms), axis = 1)



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    hist, edges = np.histogram(wordmap.ravel(), bins = dict_size, density = False)
    
    return hist/hist.sum()



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3) (as given in the write-up)
    '''
    #curating weights for layers
    weight = []
    weight.append(1)
    weight.append(2**(-1))
    for i in range(2, layer_num+1):
        weight.append(1/2**(-layer_num + i +1))
    
    #iterating over layers
    hist_image = np.array([])
    for layer in range(layer_num):
        index = 2**layer
        
        # splitting into cells and computing histograms for each layer
        split_vert = np.array_split(wordmap, index, axis = 0 )
        for i in split_vert:
            split_hor =  np.array_split(i, index, axis = 1) 
            for j in split_hor:
                hist_cell = get_feature_from_wordmap(j, dict_size)
                hist_image = np.append(hist_image, weight[layer]*hist_cell)
   
    return hist_image/np.sum(hist_image)






    

