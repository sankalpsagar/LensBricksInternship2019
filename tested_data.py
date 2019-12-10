import cv2 
import numpy as np
import os
import argparse
from random import randint,uniform
import imutils
import json
from math import *
import collections

# python3 tested_data.py --back /full/path/to/Images/ --input /full/path/to/databaseOcto/ --output /full/path/to/JPEGImages/ --outputlabels /full/path/to/labels/

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--back", required=True,
	help="path to input backgrounds")
ap.add_argument("-i", "--input", required=True,
	help="path to input images")
ap.add_argument("-o", "--output", required=True,
	help="path to output images")
ap.add_argument("-ol", "--outputlabels", required=True,
	help="path to output labels")
args = vars(ap.parse_args())
box_dict = {}
img_list = []
down_img = args['back']
database_img = args['input']
save_path = args['output']
save_path_txt = args['outputlabels']

for img in os.listdir(database_img):
	img_list.append(img)

# print(len(img_list))


def rotTranslation(image,start_valx,start_valy,s_rows,s_cols):
	rows,cols,ch = image.shape
	rand_num = randint(300,600)
	
	matrix = cv2.getRotationMatrix2D((cols/2,rows/2),rand_num,1.0)
	
	imageRotated = cv2.warpAffine(image,matrix,(cols,rows),
		borderMode =cv2.BORDER_CONSTANT,borderValue = (255,255,255)) 
	
	thirdv = np.array([start_valy,start_valx,1])
	
	fourv = np.array([start_valy+s_rows,start_valx,1]) 
	fivev = np.array([start_valy+s_rows,start_valx+s_cols,1])
	sixv = np.array([start_valy,start_valx+s_cols,1])
	
	result1 =np.dot(matrix,thirdv)
	result2 =np.dot(matrix,fourv)
	result3 =np.dot(matrix,fivev)
	result4 =np.dot(matrix,sixv)
	
	return imageRotated,result1.astype('int'),result2.astype('int'),result3.astype('int'),result4.astype('int')
def affineTransform(image,start_valx,start_valy,s_rows,s_cols):
	rows,cols,ch = image.shape
	pts1 = np.float32([[24,70],[363,131],[8,229]])
	pts2 = np.float32([[50,150],[370,180],[30,300]])
	matrix = cv2.getAffineTransform(pts1,pts2)

	affinedimage = cv2.warpAffine(image,matrix,(cols,rows),
		borderMode =cv2.BORDER_CONSTANT,borderValue = (255,255,255))

	thirdv = np.array([start_valy,start_valx,1])
	fourv = np.array([start_valy+s_rows,start_valx,1]) 
	fivev = np.array([start_valy+s_rows,start_valx+s_cols,1])
	sixv = np.array([start_valy,start_valx+s_cols,1])

	result1 =np.dot(matrix,thirdv)
	result2 =np.dot(matrix,fourv)
	result3 =np.dot(matrix,fivev)
	result4 =np.dot(matrix,sixv)

	return affinedimage,result1.astype('int'),result2.astype('int'),result3.astype('int'),result4.astype('int')


def perspectTransform(image,start_valx,start_valy,s_rows,s_cols):
	rows,cols,ch = image.shape
	pts1 = np.float32([[65,95],[345,96],[25,373],[372,401]])
	pts2 = np.float32([[45,80],[325,90],[21,354],[360,381]])
	matrix = cv2.getPerspectiveTransform(pts1,pts2)

	perspectedimage = cv2.warpPerspective(image,matrix,(cols,rows),
		borderMode =cv2.BORDER_CONSTANT,borderValue = (255,255,255))

	thirdv = np.array([start_valy,start_valx,1])
	fourv = np.array([start_valy+s_rows,start_valx,1]) 
	fivev = np.array([start_valy+s_rows,start_valx+s_cols,1])
	sixv = np.array([start_valy,start_valx+s_cols,1])

	result1 =np.dot(matrix,thirdv)
	result2 =np.dot(matrix,fourv)
	result3 =np.dot(matrix,fivev)
	result4 =np.dot(matrix,sixv)

	result1 = result1[:-1]
	result2 = result2[:-1]
	result3 = result3[:-1]
	result4 = result4[:-1]

	return perspectedimage,result1.astype('int'),result2.astype('int'),result3.astype('int'),result4.astype('int')


def scaling(image):
	num =round(uniform(0.2,0.9),2)
	scaledimage = cv2.resize(image,None,fx=num,fy = num,interpolation = cv2.INTER_AREA)
	return scaledimage

count =0 
for d_img in os.listdir(down_img):
	background = cv2.imread(down_img+d_img)
	flip_list = [0,1,-1]
	flip_rand = randint(0,2)
	background = cv2.flip(background,flip_list[flip_rand])
	background = cv2.resize(background,(416,416))
	background =  background.astype(float)
	skip_val = randint(300, 400)
	for img in range(0,len(img_list),skip_val):	
		foreground = cv2.imread(database_img+img_list[img])
		foreground = cv2.cvtColor(foreground,cv2.COLOR_BGR2RGB)
		foreground = cv2.resize(foreground,(416,416))
		
		paddingImage = np.zeros((416,416,3),np.uint8)
		paddingImage[:] = 255
		scaledimage = scaling(foreground)
	
		s_rows,s_cols,s_ch = scaledimage.shape

		start_valx = randint(min(50, 416-s_rows-25),416-s_rows-15)
		start_valy =randint(min(50, 416-s_rows-25),416-s_cols-15)
		paddingImage[start_valx:start_valx+s_rows,start_valy:start_valy+s_cols] = scaledimage
		
		myfunc = [
			rotTranslation(paddingImage,start_valx,start_valy,s_rows,s_cols),
			perspectTransform(paddingImage,start_valx,start_valy,s_rows,s_cols),
			affineTransform(paddingImage,start_valx,start_valy,s_rows,s_cols)]
		randfunc_num = randint(0,2)

		paddingImage,(x1,y1),(x2,y2),(x3,y3),(x4,y4) = myfunc[randfunc_num]

		paddingImage_gray = cv2.cvtColor(paddingImage,cv2.COLOR_BGR2GRAY)
		contours, _ = cv2.findContours(255 - paddingImage_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		area = { }
		for contour in contours:
			area[cv2.contourArea(contour)] = cv2.boundingRect(contour)
	
		area_sorted = collections.OrderedDict(sorted(area.items()))
		
		area_sorted_list = list(area_sorted)
		# print(area_sorted)
		if len(area_sorted_list) >= 1:
			req = area_sorted_list[-1]

			for key,value in area_sorted.items():
				if req == float(key):
					req_bound = value
					break
			
			x,y,w,h = req_bound
			# cv2.rectangle(paddingImage,(x,y),(x+w,y+h),(255,0,0),2)
			# cv2.imshow('paddingImage',paddingImage)

		
			paddingImage =  paddingImage.astype(float)

			alpha = paddingImage/255.0

			paddingImage = cv2.multiply(1.0 - alpha,paddingImage)
			background2 = cv2.multiply(alpha,background)

			out_image = cv2.add(background2,paddingImage)

			x,y,w,h = x/416,y/416,w/416,h/416
			# cv2.imshow('frame',out_image.astype('uint8'))
			# cv2.waitKey(0)
			file_path = str(randint(1, 900)) + d_img[::-4] + img_list[img][::-4]
			cv2.imwrite(save_path+file_path+'.jpg',out_image)
			# box_dict[d_img[::-4]+img_list[img][::-4]] = {'x':x,'y':y,'w':w,'h':h}
			box_dict[d_img[::-4]+img_list[img][::-4]] = {'x_centre':x+(w/2),'y_centre':y+(h/2),'w':w,'h':h}

			str_list =[str(0),str(x+(w/2)),str(y+(h/2)),str(w),str(h)]
			with open(save_path_txt+file_path+'.txt','w+') as txtfile:
				txtfile.write(' '.join(str_list))
				# print(txtfile)
			txtfile.close()


			#with open('JSON_DATA.json','w') as json_file:
			#	json.dump(box_dict,json_file,indent=2) 
			#json_file.close()
			# print(box_dict)
			
		else:
			pass
	count +=1
	print(count)
	
