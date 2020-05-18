

# Insert your package here
import numpy as np
import helper
import scipy.ndimage
import cv2
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    M = float(M)
    pts1 = pts1/M
    pts2 = pts2/M
    
    # curate A matrix
    A = np.zeros((len(pts1), 9))
    A[:,0] = pts1[:,0] * pts2[:,0]
    A[:,1] = pts1[:,0] * pts2[:,1]
    A[:,2] = pts1[:,0] 
    A[:,3] = pts1[:,1] * pts2[:,0]
    A[:,4] = pts1[:,1] * pts2[:,1]
    A[:,5] = pts1[:,1] 
    A[:,6] = pts2[:,0] 
    A[:,7] = pts2[:,1] 
    A[:,8] = np.ones(len(pts1))
    
    # perform SVD, the last row of V is the eigen vector corresponding to the least eigen value
    S, D, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    # enforce the singularity condition 
    F = helper._singularize(F)
    F = helper.refineF(F, pts1, pts2)
    
    # unscale the fundamental matrix
    T = np.diag((1.0/M, 1.0/M, 1.0))
    F = (np.transpose(T).dot(F)).dot(T)  
    np.savez('../results/q2_1.npz', F = F, M = M)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    M = float(M)
    pts1 = pts1/M
    pts2 = pts2/M
    
    # curate A matrix
    A = np.zeros((len(pts1), 9))
    A[:,0] = pts1[:,0] * pts2[:,0]
    A[:,1] = pts1[:,0] * pts2[:,1]
    A[:,2] = pts1[:,0] 
    A[:,3] = pts1[:,1] * pts2[:,0]
    A[:,4] = pts1[:,1] * pts2[:,1]
    A[:,5] = pts1[:,1] 
    A[:,6] = pts2[:,0] 
    A[:,7] = pts2[:,1] 
    A[:,8] = np.ones(len(pts1))
    
    # perform SVD, the last row of V is the eigen vector corresponding to the least eigen value
    S, D, V = np.linalg.svd(A)
    F0 = V[-1].reshape(3,3)
    F1 = V[-2].reshape(3,3)
    F = lambda a:np.linalg.det(a*F0 + (1-a)*F1)
    
    C1 = 0.5*(F(1)-F(-1))-2*(F(1)-F(-1))/3-(F(2)-F(-2))/12
    C2 = 0.5*F(1)+0.5*F(-1)-F(0)
    C3 = 2*(F(1)-F(-1))/3 - (F(2)-F(-2))/12
    C4 = F(0)
    Farray = []
    root = np.roots(np.asarray([C1,C2,C3,C4]))
    T = np.diag((1.0/M, 1.0/M, 1.0))
    for i in root:
        a = float(np.real(i))
        FN = a*F0 + (1-a)*F1
        FN = (np.transpose(T).dot(FN)).dot(T)  
        Farray.append(FN)
        
    np.savez('../results/q2_2.npz', F=Farray, M=M, pts1=pts1, pts2=pts2)
    return Farray





'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = (np.transpose(K2).dot(F)).dot(K1)  
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    
    A = np.zeros((4,4))
    P = np.zeros((len(pts1), 4))
    proj_pts1 = np.zeros((len(pts1), 2))
    proj_pts2 = np.zeros((len(pts1), 2))
    for i in range(len(pts1)):
        A[0,:] = pts1[i,1] * C1[2,:]-C1[1,:]
        A[1,:] = pts1[i,0] * C1[2,:]-C1[0,:]
        A[2,:] = pts2[i,1] * C2[2,:]-C2[1,:]
        A[3,:] = pts2[i,0] * C2[2,:]-C2[0,:]
        
        S, D, V = np.linalg.svd(A)
        f = V[-1]
        
        P[i,:] = f/f[-1]
    
    proj_1 = (C1.dot(P.T)).T
    proj_2 = (C2.dot(P.T)).T
    
    proj_pts1[:, 0] = np.divide(proj_1[:,0],proj_1[:,-1])
    proj_pts1[:, 1] = np.divide(proj_1[:,1],proj_1[:,-1])
    proj_pts2[:, 0] = np.divide(proj_2[:,0],proj_2[:,-1])
    proj_pts2[:, 1] = np.divide(proj_2[:,1],proj_2[:,-1])
    
    error = np.sum((proj_pts1 - pts1)**2) + np.sum((proj_pts2 - pts2)**2 )
    
    return P[:, :3], error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

def epipolarEndpoints(x1, y1, F, I1, I2):
    sy, sx, _ = I2.shape
    v = np.array([x1, y1, 1])
    l = F.dot(v)
    s = np.sqrt(l[0]**2+l[1]**2)
    
    if s == 0:
        print('ERROR: Zero line vector in displayEpipolar')
        return 0, 0, 0, 0

    l = l/s

    if l[0] != 0:
        ye = sy-1
        ys = 0
        xe = -(l[1] * ye + l[2])/l[0]
        xs = -(l[1] * ys + l[2])/l[0]
    else:
        xe = sx-1
        xs = 0
        ye = -(l[0] * xe + l[2])/l[1]
        ys = -(l[0] * xs + l[2])/l[1]
        
        
    return xe, xs, ye, ys
def epipolarCorrespondence(im1, im2, F, x1, y1):
    
    xs, xe, ys, ye = epipolarEndpoints(x1, y1, F, im1, im2)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    h = im1.shape[0]
    w = im1.shape[1]
    errors = []
    pw = 25
    p = 51
    
    # create a square patch around the x1 and y1 points of image1
    x = np.linspace(x1-pw, x1+pw, p, dtype=int)
    y = np.linspace(y1-pw, y1+pw, p, dtype=int)
    X, Y = np.meshgrid(x, y)
    B1 = scipy.ndimage.filters.gaussian_filter(gray1[Y, X], 2)
    
    # find points on epipolar line, y values have the greatest variation
    dy = abs(ye - ys)
    #dx = abs(int(xe) - int(xs))
    xx = int((xe+xs)/2)
    #xx = np.linspace(int(xs), int(xe), dx)
    yy = np.linspace(ys, ye, dy)
    
    # loop over every point on the epipolar line of image 2 and create patches for each. If the lines are not parallel put an x loop
    #for i in range(dx):
    for j in range(dy):
            xxx = np.linspace(xx-pw, xx+pw, p, dtype = int)
            yyy = np.linspace(yy[j]-pw, yy[j]+pw, p, dtype = int)
            #check whether this patch lies within the image dimensions
            xxx[xxx < 0] = 0
            xxx[xxx >= w] = w-1
            yyy[yyy < 0] = 0
            yyy[yyy >= h] = h-1
            x1, y1 = np.meshgrid(xxx, yyy)
            B2 = scipy.ndimage.filters.gaussian_filter(gray2[y1, x1], 2)
            error = np.square(np.linalg.norm(B1 - B2))
            errors.append(error)
            #index.append((i,j))
    i = np.argmin(errors)
    #m, n = index[i]
    return xx, yy[i]

    
if __name__=='__main__':
   


    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')

    data_corr = np.load('../data/some_corresp.npz')
    pts1 = data_corr['pts1']
    pts2 = data_corr['pts2'] 

    data = np.load('../data/intrinsics.npz')
    K1 = data['K1']
    K2 = data['K2']

    image_dim = [im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]]
    M = max(image_dim)


    #2.1
    F1 = eightpoint(pts1, pts2, M)
    #helper.displayEpipolarF(im1, im2, F1)

    #2.2
    rand_index = [82, 19, 56, 84, 54, 24, 18]
    p1 = np.zeros((7,2))
    p2 = np.zeros((7,2))
    for i in range(len(rand_index)):
    
        p1[i] = pts1[rand_index[i]]
        p2[i] = pts2[rand_index[i]]

    F2 = sevenpoint(p1, p2, M)
    
    #helper.displayEpipolarF(im1, im2, F2[2])
    
    E = essentialMatrix(F1, K1, K2)
    np.savez('../results/q4_1.npz', F = F1, pts1 = pts1, pts2 =pts2)
    helper.epipolarMatchGUI(im1,im2,F1)

    
    
'''
error = 1000
ri = 0
FF = np.array([])

for j in range(10000):
    
    rand_index = np.random.choice(len(pts1), 7, replace = False)
    p1 = np.zeros((7,2))
    p2 = np.zeros((7,2))
    for i in range(len(rand_index)):
    
        p1[i] = pts1[rand_index[i]]
        p2[i] = pts2[rand_index[i]]

    F2 = submission.sevenpoint(p1, p2, M)
    
    
    for i in F2:
        
        #helper.displayEpipolarF(im1, im2, i)
        MSE = np.square(np.subtract(i,F1)).mean() 
        
        if MSE<error:
            error = MSE
            FF = i
            ri = rand_index
print(error)            
print(ri) 
print(p1, p2)
print(FF)


'''
    
 




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    