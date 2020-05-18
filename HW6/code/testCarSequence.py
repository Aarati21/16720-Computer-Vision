import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanade import LucasKanade

carseq = np.load('../data/carseq.npy')
h, w, f = carseq.shape
rect = np.array([59.0, 116.0, 145.0, 151.0])
car_rects = rect

i = 1
plt.figure(figsize = (48,10))

for j in range(1, f):
    
    It = carseq[:,:,j-1]
    It1 = carseq[:,:,j]
    p = LucasKanade(It, It1, rect)
   
    rect += np.array([p[0],p[1],p[0],p[1]])
    
    car_rects = np.vstack((car_rects, rect))
    
    if j in [1, 100, 200, 300, 400]:
        plt.subplot(1, 5, i)
        i+=1
        plt.axis('off')
        plt.imshow(carseq[:,:,j], cmap = 'gray')
        plt.gca().add_patch(patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth = 3, edgecolor = 'r', facecolor = 'none'))
        plt.subplots_adjust(wspace=0.05)

car_rects = car_rects.reshape(-1, 4)
np.save('../results/carseqrects.npy', car_rects)
plt.show()

