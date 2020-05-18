import numpy as np

    
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    im_warped = np.zeros((output_shape[0], output_shape[1]))
    for i in range(output_shape[0]):
      for j in range(output_shape[1]):
        p_warped = np.array([i, j, 1])
        p_source = np.linalg.inv(A).dot(p_warped.T)
        p_source = [round(i) for i in p_source]
        p_source = [int(i) for i in p_source]
        if (p_source[0] > 0 and p_source[0] < output_shape[0]-1 and p_source[1] > 0 and p_source[1] <output_shape[1]-1):
          im_warped[i][j] = im[p_source[0]][p_source[1]]
    return im_warped

