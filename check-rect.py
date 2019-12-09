# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#    help="path to input image")
args = vars(ap.parse_args())
 
######## for the camera input 
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# cap = cv2.VideoCapture(0)
 
# while True:
#     # Getting out image by webcam 
#     _, image = cap.read()
#     # Converting the image to gray scale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#     # Get faces into webcam's image
#     rects = detector(gray, 0)
    
#     # For each detected face, find the landmark.
#     for (i, rect) in enumerate(rects):
#         # Make the prediction and transfom it to numpy array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
    
#         # Draw on our image, all the finded cordinate points (x,y) 
#         for (x, y) in shape:
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
#     # Show the image
#     cv2.imshow("Output", image)
    
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()
# cap.release()


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
 

# detect faces in the grayscale image
# rects = detector(gray, 1)
rects =[[2,3,4,5]]
# loop over the face detections
cv2.namedWindow('clone',cv2.WINDOW_NORMAL)
'''for m in range(1,16):
    if m<=10:
        member_path  = '/media/hugeDrive/selfie_styler/selfiemorph/Members_auto/Member {}/Member Images/Member {}.png'.format(m,m)
    else:
        member_path ='/media/hugeDrive/selfie_styler/selfiemorph/new_members_auto/Members/Member{}/Member{}.png'.format(m,m)'''
member_path = '/media/arshad/hugeDrive/Sankalp/test-data-3/TRIAL_PIC.png'
    # load the input image, resize it, and convert it to grayscale
print(member_path)
image = cv2.imread(member_path,cv2.IMREAD_UNCHANGED)

    # image = imutils.resize(image, width=500)
'''mask = image[...,3]
mask = np.where(mask,255,0).astype('uint8')
cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cnts  =  sorted(cnts,key=cv2.contourArea)
cnt = cnts[-1]
x,y,w,h = cv2.boundingRect(cnt) 

gray = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
color = [0,255,0]
vis  = gray.copy()'''
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
'''
for c in range(3):

    gray[...,c] = np.where(mask==255,gray[:,:,c],color[c]).astype('uint8')

color = [0,0,0]
for c in range(3):

    vis[...,c] = np.where(mask==255,vis[...,c],color[c]).astype('uint8')
'''

'''for (i, rect) in enumerate(rects):'''
x = 0
y = 20
w = 1020
h = 1020
rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
shape = predictor(gray, rect)

shape = face_utils.shape_to_np(shape)

        #print(shape)
print(len(shape))
     
        # loop over the face parts individually
        # for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        #     # clone the original image so we can draw on it, then
        #     # display the name of the face part on the image
clone = image.copy()
print(x,y,w,h)
print("clone image shape is ",clone.shape)
        #print(type(shape))
        #cv2.rectangle(gray,(x-150,y+100),(x+w+150,1280),(0,255,0),3)
        #     cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #         0.7, (0, 0, 255), 2)
     
        #     # loop over the subset of facial landmarks, drawing the
        #     # specific face part

np.save(member_path.split('.')[0] + '.npy',shape)
        
for (x, y) in shape:
            #print(x, y)
    cv2.circle(gray, (x, y), 1, (0, 0, 255),5)
        #cv2.imshow("clone"ray)
            # extract the ROI of the face region as a separate image
        # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # roi = image[y:y + h, x:x + w]
        # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
     
            # # show the particular face part
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", clone)
            # cv2.waitKey(0)
     
        # visualize all facial landmarks with a transparent overlay
        # output = face_utils.visualize_facial_landmarks(image, shape)
        # cv2.imshow("Image", output)
        #cv2.waitKey(0)
    '''cv2.imwrite(os.path.join('outs_v4',os.path.basename(member_path)),vis[:,:,::-1])'''
cv2.imshow("Image", gray)
cv2.waitKey(0)