from PIL import Image
import os.path, sys

a = []

with open("out.txt", "r") as f:
    lines = f.readlines()

    # Loop through all lines.
    for line in lines:
       s = line.split("\t")
       print(s)
       a.append([s[0], s[1], s[2], s[3]])

print (a)

new_path = '/media/arshad/hugeDrive/Sankalp/test-data-5/'
items = os.listdir(new_path)
count=0
number=0
img_list = []

for names in sorted(items):
    if names.endswith(".jpg") or names.endswith(".JPG"):
        img_list.append(names)
print(img_list)

'''
for i in range(len(a)):
    print("bruh")
    if a[0][i]!=img_list[i]:
        print("EW")
        break
'''

for l in range(len(img_list)):
        print(l)
        f = img_list[l]
        print(number)
        print(new_path + img_list[l])
        im = Image.open(new_path + '/' + img_list[l])
        try:
            x = int(a[number][0])*0.95
            y = int(a[number][1])*0.95
            w = int(a[number][2])*1.1
            h = int(a[number][3])*1.1
            print(x, y, w, h)
        except:
            print("Index outta range I guess")
        im1 = im.crop((x, y, w+x, h+y))
        #im1.show()
        im1 = im1.save("/media/arshad/hugeDrive/Sankalp/Cropped-Test/Cropped"+f)
        number+=1
