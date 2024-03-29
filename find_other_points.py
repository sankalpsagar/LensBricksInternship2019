


import numpy as np
import cv2
import argparse
import os

w = 1024
h = 1024

# Check stack overflow for detailed algorithm
def createLineIterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)                  
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = np.float32(dX) / np.float32(dY)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = np.float32(dY) / np.float32(dX)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[np.uint(itbuffer[:,1]), np.uint(itbuffer[:,0])]

    return itbuffer

def midpoint (x1, y1, x2, y2):
    return (int((x1+x2)/2), int((y1+y2)/2))

# get all the files in the folder
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input images")
args = vars(ap.parse_args())
img_list = []

database_img = args['input']
for img in sorted(os.listdir(database_img)):
    img_list.append(img)

# print(img_list)
fimgflag = True

# testing purposes
# img_list = ["IMG_000.png"]

print("<?xml version='1.0' encoding='ISO-8859-1'?>")
print("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>")
print("<dataset>")
print("<name>octoshape dataset generated by find_other_points.py</name>")
print("<images>", end = '')

#1-5
red = (0, 0, 255)
black = (0, 0, 0)
lime = (0, 255, 0)
blue = (255, 0, 0)
yellow = (255, 255, 0)
        
#6-10
cyan = (255, 255, 0)
magneta = (255, 0, 255)
silver = (192, 192, 192)
grey = (128, 128, 128)
maroon = (0, 0, 128)
        
#11-15
olive = (0, 128, 128)
green = (0, 128, 0)
purple = (128, 0, 128)
teal = (128, 128, 0)
navy = (128, 0, 0)

#16-20
gold = (0, 215, 255)
indigo = (130, 0, 75)
pink = (147, 20, 255)
brown = (19, 69, 139)
sblue = (255, 144, 30)

#21-25
dgray = (79, 79, 47)
sgreen = (87, 139, 46)
fgreen = (34, 139, 34)
orange = (0, 69, 255)
mviolet = (133, 21, 199)

count = 0

for img in img_list:
    if img.endswith('.PNG') or img.endswith('.png'):
        if fimgflag == True:
            print("<image file='", img ,"'>", sep = '' )
            fimgflag = False
        else:
            print("\t<image file='", img ,"'>", sep = '' )
        img = cv2.imread(img)

        # image=np.zeros((h,w,3),np.uint8)
        image = np.zeros([h, w, 3], dtype=np.uint8)
        image.fill(255)
        # print(red)
        print("\t\t<box top='179' left='179' width='663' height='673'>")
        # so third chanel is present
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        itbuf = createLineIterator((511,199), (731,291), gray)

        # print(itbuf)
        print("\t\t\t<part name='0' x='{}' y='{}'/>".format(511, 199))
        image[199, 511]=red
        # print("Point 0: 511 , 199")
        img = cv2.circle(img, (511 , 199), 1, (0, 0, 255), 2)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='1' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=black
                # print("Point 1:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (731, 291, int(x), (y))
                print("\t\t\t<part name='2' x='{}' y='{}'/>".format(a, b))
                image[b, a]=lime
                # print("Point 2:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        # cv2.imshow('idk', image)
        # cv2.waitKey(0)
        print("\t\t\t<part name='3' x='{}' y='{}'/>".format(731, 291))
        image[731, 291]=blue
        # print("Point 3: 731 , 291")
        img = cv2.circle(img, (731 , 291), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((731, 291), (824, 511), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='4' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=yellow
                # print("Point 4:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (824, 511, int(x), (y))
                print("\t\t\t<part name='5' x='{}' y='{}'/>".format(a, b))
                image[b, a]=cyan
                # print("Point 5:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='6' x='{}' y='{}'/>".format(824, 511))
        image[511, 824]=magneta
        # print("Point 6: 824 , 511")
        img = cv2.circle(img, (824 , 511), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((824, 511), (731, 735), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='7' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=silver
                # print("Point 7:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (731, 735, int(x), (y))
                print("\t\t\t<part name='8' x='{}' y='{}'/>".format(a, b))
                image[b, a]=grey
                # print("Point 8:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='9' x='{}' y='{}'/>".format(731, 735))
        image[735, 731]=maroon
        # print("Point 9: 731 , 735")
        img = cv2.circle(img, (731 , 735), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((731, 735), (510, 826), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='10' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=olive
                # print("Point 10:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (510, 826, int(x), (y))
                print("\t\t\t<part name='11' x='{}' y='{}'/>".format(a, b))
                image[b, a]=green
                # print("Point 11:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='12' x='{}' y='{}'/>".format(510, 826))
        image[826, 510]=purple
        # print("Point 12: 510 , 826")
        img = cv2.circle(img, (510 , 826), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((510, 826), (290, 734), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='13' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=teal
                # print("Point 13:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (290, 734, int(x), (y))
                print("\t\t\t<part name='14' x='{}' y='{}'/>".format(a, b))
                image[b, a]=navy
                # print("Point 14:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='15' x='{}' y='{}'/>".format(290, 734))
        image[734, 290]=gold
        # print("Point 15: 290 , 734")
        img = cv2.circle(img, (290 , 734), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((290, 734), (198, 514), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='16' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=indigo
                # print("Point 16:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (198, 514, int(x), (y))
                print("\t\t\t<part name='17' x='{}' y='{}'/>".format(a, b))
                image[b, a]=pink
                # print("Point 17:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='18' x='{}' y='{}'/>".format(197, 513))
        image[513, 197]=brown
        # print("Point 18: 197 , 513")
        img = cv2.circle(img, (198 , 514), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((198, 514), (289, 293), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='19' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=sblue
                # print("Point 19:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (289, 293, int(x), (y))
                print("\t\t\t<part name='20' x='{}' y='{}'/>".format(a, b))
                image[b, a]=dgray
                # print("Point 20:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='21' x='{}' y='{}'/>".format(289, 293))
        image[293, 289]=sgreen
        # print("Point 21: 289, 293")
        img = cv2.circle(img, (289, 293), 1, (0, 0, 255), 2)

        itbuf = createLineIterator((289, 293), (511,199), gray)

        for x, y, z in itbuf:
            if z == 0:
                print("\t\t\t<part name='22' x='{}' y='{}'/>".format(int(x), int(y)))
                image[int(y), int(x)]=fgreen
                # print("Point 22:",x, ",",y)
                img = cv2.circle(img, (int(x) , int(y)), 1, (0, 0, 255), 2)
                a, b = midpoint (511,199, int(x), (y))
                print("\t\t\t<part name='23' x='{}' y='{}'/>".format(a, b))
                image[b, a]=orange
                # print("Point 23:",a,",",b)
                img = cv2.circle(img, (a , b), 1, (0, 0, 255), 2)
                break

        print("\t\t\t<part name='24' x='{}' y='{}'/>".format(511, 511))
        image[511,511]=mviolet
        print("\t\t</box>")
        print("\t</image>")
        # print("Point 24: 511 , 511")
        img = cv2.circle(img, (511 , 511), 1, (0, 0, 255), 2)
        path = '/media/arshad/hugeDrive/Sankalp/yaynewprob/points'
        # cv2.imshow('idk', image)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(path , 'IMG_'+str(count)+'_points.PNG'), image)
        # color = image[293][289]
        # print(color)
        count+=1

        # testing purposes
        if count == 1:
            break
        # testing purposes
        # cv2.imshow('final marked', img)
        # cv2.waitKey(0)
print("</images></dataset>")

# img = cv2.imread('/media/arshad/hugeDrive/Sankalp/yaynewprob/points/IMG_0_points.PNG')

# lower_red = red   BGR-code of your lowest red
# upper_red = red   BGR-code of your highest red 
# mask = cv2.inRange(image, lower_red, upper_red)  

# coord=cv2.findNonZero(mask)
# print(coord)