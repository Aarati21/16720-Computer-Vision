import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

airseq = np.load('../data/aerialseq.npy')
h, w, f = airseq.shape

fig, axes = plt.subplots(1,4, figsize=(48, 10))
j = 0

for i in range(1, f):
    
    It = airseq[:, :, i-1]
    It1 = airseq[:,:,i]
    mask = SubtractDominantMotion(It, It1)
    
    image = np.zeros((h, w, 3))
    image[:,:,0] = It1
    image[:,:,1] = It1
    image[:,:,2] = It1
    image[:,:,2][mask == 1] = 1
    
    if i in [30, 60, 90, 120]:
        
        axes[j].imshow(image)
        axes[j].set_title("Frame = " + str(i))
        j += 1

    