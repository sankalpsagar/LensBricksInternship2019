


import numpy as np 
from matplotlib import pyplot as plt 
import cv2

import glob
import cv2 as cv

path = glob.glob("/media/arshad/hugeDrive/Sankalp/test-data/*.png")
cv_img = []
num = 1
for i in path:
# opencv loads the image in BGR, convert it to RGB
	img = cv2.cvtColor(cv2.imread(i),
                   cv2.COLOR_BGR2RGB)
	lower_white = np.array([220, 220, 220], dtype=np.uint8)
	upper_white = np.array([255, 255, 255], dtype=np.uint8)
	mask = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
	mask = cv2.bitwise_not(mask)  # invert mask

	# load background (could be an image too)
	bk = np.full(img.shape, 255, dtype=np.uint8)  # white bk

	# get masked foreground
	fg_masked = cv2.bitwise_and(img, img, mask=mask)

	# get masked background, mask must be inverted 
	mask = cv2.bitwise_not(mask)
	bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

	# combine masked foreground and masked background 
	final = cv2.bitwise_or(fg_masked, bk_masked)
	mask = cv2.bitwise_not(mask)  # revert mask to original

#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
	
	cv2.imwrite("/media/arshad/hugeDrive/Sankalp/test-data/IMG"+str(num)+".png", img)
	num +=1
print("Done.")