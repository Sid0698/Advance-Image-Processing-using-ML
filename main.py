import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os


from utils1 import *
from modules import *

print('Choose Operation:\n')
operation = input(' 0:Convert to grayscale \n 1:Sobel Operator \n 2:Gaussian Filter \n 3:histrogram equilization\n 4:Otsu Thresholding\n 5:Erosion\n 6:Dilation\n 7:Opening\n 8:Closing\n ')
#for k, v in op_dict.items():
#	print(k, v)
img_rgb= np.asarray(Image.open('fort.jpg'))

while(1):
	# operation = int(input('\nEnter operation you wish to perform:'))

	if operation=='0':	
		img_gray = rgb2gray(img_rgb)
		plt.imshow(img_gray)

	if operation=='1':
		sobel(img_rgb)

	if operation=='2':
		scaling(img_gray)

	if operation == '3':
		hest(img_gray)
	
	if operation == '4':
		th=otsu(img_gray)

	if operation == '5':
		erosion(th)

	if operation == '6':
		dilation(th)	

	if operation == '7':
		open_img=opening(th)
		plt.imshow(open_img)

	if operation == '8':
		close_img=closing(th)
		plt.show(close_img)
