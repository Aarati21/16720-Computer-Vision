import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from LucasKanade import LucasKanade

carseq = np.load('../data/carseq.npy')
car_rects1 = np.load('../results/carseqrects.npy')
h, w, f = carseq.shape
rect = np.array([59.0, 116.0, 145.0, 151.0])
rect0 = np.array([59.0, 116.0, 145.0, 151.0])
car_rects = []
car_rects.append(rect)
i = 1
p1 = np.zeros(2)
plt.figure(figsize = (48,10))
p2 = np.zeros(2)
for j in range(1, f):
    
    It = carseq[:,:,j-1]
    It1 = carseq[:,:,j]
    p = LucasKanade(It, It1, rect, p2)
    p1 += p
    pnew = LucasKanade(carseq[:,:,0], It1, rect0, p1)
    
    
    if (np.linalg.norm(pnew-p-p1,1)) >= 1:
        rect += np.array((p[0],p[1],p[0],p[1]))
        
    else:
        rect = rect0 + np.array((pnew[0],pnew[1],pnew[0],pnew[1]))
        p = pnew - p1
    car_rects.append(rect)
    p2 = p

    if j in [1, 100, 200, 300, 400]:
        plt.subplot(1, 5, i)
        i+=1
        plt.axis('off')
        plt.imshow(carseq[:,:,j], cmap = 'gray')
        plt.gca().add_patch(patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth = 3, edgecolor = 'r', facecolor = 'none'))
        plt.gca().add_patch(patches.Rectangle((car_rects1[j][0], car_rects1[j][1]), car_rects1[j][2] - car_rects1[j][0], car_rects1[j][3] - car_rects1[j][1], linewidth = 3, edgecolor = 'b', facecolor = 'none'))
        plt.subplots_adjust(wspace=0.05)
        
        

rects = np.array(car_rects)
rects = rects.reshape(-1, 4)
np.save('../results/carseqrects-wcrt.npy', rects)
plt.show()




