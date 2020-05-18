import numpy as np

def ssd(color1, color2):
    ssd_min = np.sum((color1-color2)**2)
    
    for i in range(-30, 30):
        for j in range(-30, 30):
            color_shift = np.roll(color2, i, 0)
            color_shift = np.roll(color_shift, j, 1)
            ssd_new = np.sum((color1-color_shift)**2)
            
            if ssd_new < ssd_min:
                ssd_min = ssd_new
                i_minshift = i
                j_minshift = j
                
    color2_new = np.roll(color2, i_minshift, 0) 
    color2_new = np.roll(color2_new, j_minshift, 1)      
    return color2_new
            
            
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    H = red.shape[0]
    W = red.shape[1]
    out = np.zeros((H, W, 3), 'uint8')
    out[:,:,0] = red
    out[:,:,1] = ssd(red, green)
    out[:,:,2] = ssd(red, blue)
    
    return out