import cv2
import numpy as np 
import matplotlib.pyplot as plt 
#import pdb
#mport argparse


def myHough(img_name, ce_params, hl_params): 
    img = cv2.imread(img_name)
    #print(img.shape[0], img.shape[1])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blur,ce_params[0], ce_params[1], ce_params[2])
    
    
    lines = cv2.HoughLinesP(edges,rho =hl_params[0], theta =hl_params[1], threshold = hl_params[2], minLineLength=hl_params[3], maxLineGap=hl_params[4])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        cv2.line(img,(x1,y1),(x2,y2),(0,255, 0),2)
    
    cv2.imwrite('../results/img01_hlines.jpg',img)
    cv2.imshow('../results/img01_hlines.jpg',img)
    
    pass

def plot_edges(img_name, ce_params, hl_params):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,ce_params[0], ce_params[1], ce_params[2])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__=="__main__":

    # create a list of the params 
    # for both your edge detector 
    # hough transform

    edge_params = []
    hl_params = []
    img_name = "../data/img01.jpg"
                   
    ce_params = [10, 150,  3, False]
    hl_params = [800, np.pi/180, 200, 0, 0]
    plot_edges(img_name, ce_params, hl_params)
    myHough(img_name, ce_params, hl_params)
    