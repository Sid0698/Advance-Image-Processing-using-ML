import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os

def rgb2gray(img_rgb):
    plt.imshow(img_rgb)
    return np.dot(img_rgb[...,:3], [0.299, 0.587, 0.144])

def convolve(image, window):
    tmp_a = np.zeros((
        image.shape[0] + window.shape[0] // 2 * 2,
        image.shape[1] + window.shape[1] // 2 * 2), float)
    tmp_a[ window.shape[0] // 2:window.shape[0] // 2 + image.shape[0],
    window.shape[1] // 2:window.shape[1] // 2 + image.shape[1]
    ] = image
    result = np.zeros(image.shape, float)
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            result[x, y] = np.sum(
                tmp_a[x:x + window.shape[0], y:y + window.shape[1]] * window
            )
    return result

def sobel(image):
    # image=cv2.imread("fort.jpg", cv2.IMREAD_UNCHANGED)
    grayscale=rgb2gray(image)
    x_grad=grayscale.copy().astype(float)
    y_grad=grayscale.copy().astype(float)
    gradient_mag=grayscale.copy().astype(float)
    edges=grayscale.copy().astype(float)

    sumx=0
    sumy=0
    sobelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    x_grad=convolve(grayscale, sobelx)
    y_grad=convolve(grayscale, sobely)
    gradient_mag=(x_grad**2+y_grad**2)**0.5

    x_grad=(x_grad-np.amin(x_grad))*255/(np.amax(x_grad)-np.amin(x_grad))
    y_grad=(y_grad-np.amin(y_grad))*255/(np.amax(y_grad)-np.amin(y_grad))   
    gradient_mag=(gradient_mag-np.amin(gradient_mag))*255/(np.amax(gradient_mag)-np.amin(gradient_mag))

    x_grad=x_grad.astype(np.uint8)
    y_grad=y_grad.astype(np.uint8)
    gradient_mag=gradient_mag.astype(np.uint8)
    
    plt.imshow(gradient_mag, cmap=plt.get_cmap('gray'))

def hest(img_array):
    # img_array = np.asarray(Image.open('grey_fort.jpg'))
    # plt.imshow(img_array)
    histogram_array = np.bincount(img_array.flatten())
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    img_list = list(img_array.flatten())
    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

def otsu(image):
    # image = cv2.imread("grey_fort.jpg", cv2.IMREAD_UNCHANGED)
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    total = image.shape[0]*image.shape[1]
    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0
    
    for i in range(0,256):
        sumT += i * hist[i]
    
    weightB, weightF = 0, 0
    
    for i in range(0,256):
        weightB += hist[i]
        weightF = total - weightB
        if weightF == 0: # only background pixels
            break
        
        sumB += i*hist[i]
        sumF = sumT - sumB
        
        meanB = sumB/weightB
        meanF = sumF/weightF 
        
        varBetween = weightB*weightF*(meanB-meanF)**2
        if varBetween > current_max:
            current_max = varBetween
            threshold = i  
    
    th = image
    th[th>=threshold]=255
    th[th<threshold]=0
#     plt.imshow(th)
#     print(th.shape)
    cv2.imwrite('otsu.jpg',th)
    return th

def erosion(th):
    # th = cv2.imread('otsu.jpg', cv2.IMREAD_UNCHANGED)
    print(th[1][1])
    eroded = th.copy()
    kernel_size = 3
    dx, dy , dz= kernel_size//2, kernel_size//2, kernel_size//2
    r, c, x = th.shape
    for i in range(r):
        for j in range(c):
            for m in range(x):
                if th[i][j][m] == 255:
                    flag = 0
#                   print(th.shape)
                    for k in range(i-dx,i+dx+1):
                        for l in range(j-dy,j+dy+1):
                            for n in range(m-dz,m+dz+1):
                                if k in range(r) and l in range(c) and n in range(x):
                                    if th[k][l][n] == 0:
                                        flag = flag or 1
                    if flag == 1:
                        eroded[i][j][m] = 0

    plt.imshow(eroded,cmap=plt.get_cmap('gray'))
    return eroded

def dilation(th):
    # th = cv2.imread('otsu.jpg', cv2.IMREAD_UNCHANGED)
    print(th[1][1])
    dialeted = th.copy()
    kernel_size = 3
    dx, dy , dz= kernel_size//2, kernel_size//2, kernel_size//2
    r, c, x = th.shape
    for i in range(r):
        for j in range(c):
            for m in range(x):
                if th[i][j][m] == 0:
                    flag = 0
#                   print(th[1][1][1])
                    for k in range(i-dx,i+dx+1):
                        for l in range(j-dy,j+dy+1):
                            for n in range(m-dz,m+dz+1):
                                if k in range(r) and l in range(c) and n in range(x):
                                    if th[k][l][n] == 255:
                                        flag = flag or 1
                    if flag == 1:
                        dialeted[i][j][m] = 255

    plt.imshow(dialeted,cmap=plt.get_cmap('gray'))
    return dialeted
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(th)
# ax[1].imshow(dialeted)

def gaussian1D(x, mean = 0, sigma = 1):
    num = np.exp(-(((x-mean)**2) / (2.0*sigma**2)))
    den = sigma * np.sqrt(2*np.pi)
    return num / den

def gaussian(m, n, sigma = 1):
    g = np.zeros((m,n))
    m = m // 2
    n = n // 2
    for i in range(-m,m+1):
        for j in range(-n,n+1):
            den = 2.0*np.pi*(sigma**2)
            num = np.exp(-(i**2 + j**2) / (2*(sigma**2)))
            g[i+m][j+n] = num / den
    return g

def isvalid(i, j, r, c):
    if i >= r or j >= c or i < 0 or j < 0:
        return 0
    return 1

def euc_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def filter2D(image, kernel):
    r, c = image.shape
    m, n = kernel.shape
    filtered = np.zeros(image.shape)
    dx, dy = m//2, n//2
    for i in range(r):
        for j in range(c):
            psum = 0.0
            for k in range(i-dx,i+dx+1):
                for l in range(j-dy,j+dy+1):
                    if isvalid(k,l,r,c):
                        psum += image[k][l] * kernel[i-k+dx][j-l+dy]
            filtered[i][j] = psum
    return filtered

def scaling(image, sigmag = 3, k = 5):
    kernel = gaussian(k,k,sigmag)
    print("hi")
    scaled = filter2D(image,kernel)
    plt.imshow(scaled,cmap=plt.get_cmap('gray'))
    return scaled

def opening(imageO):
    eroded = erosion(imageO)
    opened = dilation(imageO)
    return opened

def closing(imageO):
    dilated = dilation(imageO)
    closed = erosion(imageO)
    return closed