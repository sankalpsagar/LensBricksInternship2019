import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import shutil
import csv
from tqdm import tqdm
import imutils
import time
import json
import tensorflow as tf
from modules.detector import Detector as DetectorCPU
from modules.binarize import Binarize
from modules.Transform import Transform
from modules.Extract import Extract
#from modules.classifier import Classifier
import modules.error_detector as error_detector

####################################################################
yolo_directory = 'yolo_data'
model_path = 'all_freezed_v155.h5'
model_path_inner = 'all_freezed_v71.h5'
detection_confidence = 0.3
nms_threshold = 0.1
error_checker = False

#if tf.test.is_built_with_cuda():
#    from detector import Detector as DectectorGPU

#    detector = DectectorGPU(yolo_directory)
#else:
detector = DetectorCPU(yolo_directory, detection_confidence, nms_threshold)

# initialised all the required classes imported from other modules
binarize = Binarize()
transform = Transform()
extract = Extract()
#classifier = Classifier(model_path, model_path_inner)


def rotateImage(image, angle, color):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=color)

    return result


with open('input_data/table_november.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

new_path = '/media/arshad/hugeDrive/Sankalp/test-data-4/'
items = os.listdir(new_path)
img_list = []
for names in sorted(items):
    if names.endswith(".jpg") or names.endswith(".JPG"):
        img_list.append(names)
def upload():

    write_image = True
    print_flag = True
    d = error_detector.create_lookup_table('input_data/LookupTableForECC.xls')
    acc_class = 0
    acc_color = 0
    perm_count = 0
    jsonShapeError = {}
    jsonColorError = {}

    for l in tqdm(range(len(img_list))):
        #if print_flag:
           # print(img_list[l])
        image_input = cv2.imread(new_path + '/' + img_list[l], 1)
        filename = img_list[l]

    # for preserving our original image
        if image_input.shape[0] < image_input.shape[1]:
            image_input = imutils.rotate_bound(image_input, 90)
        as_ratio = image_input.shape[1] / image_input.shape[0]
        target_dim = 512
        if image_input.shape[1] > image_input.shape[0]:
            target_size = (target_dim, int(target_dim / as_ratio))
        else:
            target_size = (int(target_dim * as_ratio), target_dim)
        image_input = cv2.resize(image_input, target_size, interpolation=cv2.INTER_AREA)

        image_copy = image_input.copy()
        try:
              boxes = detector.detect(image_input)
              print(boxes[0][0], "\t", boxes[0][1], "\t", boxes[0][2], "\t", boxes[0][3])
              x = int(boxes[0][0])
              y = int(boxes[0][1])
              w = int(boxes[0][2])
              h = int(boxes[0][3])
              #cv2.rectangle(image_input, (x,y), (x+w, y+h), (0, 0, 255), 15)
              #cv2.imshow("jksdjkd", image_input)
              cv2.imwrite('/media/arshad/hugeDrive/Sankalp/test-data-5/'+img_list[l], image_input)
        except:
              print("")

'''     # Binarization Processes
        detected_image_list = binarize.detect_box_list(image_copy, filename, boxes)
        binarized_image_list, num_images_list = binarize.binarize(detected_image_list)

        bi_count = 0

        for idx in range(len(binarized_image_list)):
            image = binarized_image_list[idx]
            if (image.shape[0] > image.shape[1]):
                left = right = int((image.shape[0] - image.shape[1]) / 2)
                top = bottom = 0
            else:
                top = bottom = int((image.shape[1] - image.shape[0]) / 2)
                left = right = 0

            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))
            #        cv2.imshow("im_src padded", im_src)
            #        cv2.waitKey(0)
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            convex_hull = transform.convex(image)
            imgcut = extract.cut(image, convex_hull)
            pred_list = extract.outer_extractor(image, imgcut)
            inner_shape_crop = extract.inner_shape_crop(convex_hull, imgcut, image)
            pred_list_inner = []
            for save_idx in range(4):
                if save_idx != 0:
                    inner_shape_crop = rotateImage(inner_shape_crop, 90, (255, 255, 255))
                inner_shape_crop_3channel = cv2.merge((inner_shape_crop, inner_shape_crop, inner_shape_crop))
                pred_list_inner.append(inner_shape_crop_3channel)
                # cv2.imshow('inner_shape_crop', inner_shape_crop)
                # cv2.waitKey(0)
            class_number_truth = []
            colour_number_truth = []
            for contour_name in range(8):
                perm_count += 1
                for i in range(len(data)):
                    if (img_list[l].split('.')[0] == data[i][0] and contour_name <= 7):
                        class_number_truth.append(data[i][contour_name + 1].split('-')[0])
                        colour_number_truth.append(data[i][contour_name + 1].split('-')[1][0])

            # print (class_number_truth)
            processed_pred_list = classifier.preprocess_v2(pred_list)
            processed_pred_list_inner = classifier.preprocess_v2(pred_list_inner)
            shapes = classifier.infer(processed_pred_list, d)
            colors_pred = classifier.infer_color(processed_pred_list_inner)
            colors = []
            for idx in colors_pred:
                c1, c2 = extract.ContourNtoColors(int(idx))
                colors.append(str(c1))
                colors.append(str(c2))
            if print_flag:
                print('pred shape: ', shapes)
                print('GT shape: ', class_number_truth)
                print('-------------------------------------------------')
                print('pred color: ', colors)
                print('GT Color: ', colour_number_truth)
                print('')
                print('pred color: ', colors_pred)
                print('GT Color: ', error_detector.get_prediction(colour_number_truth))
                print('-------------------------------------------------')
            temp = 0
            temp_color = 0
            for i in range(len(pred_list)):
                if shapes[i] == class_number_truth[i]:
                    acc_class += 1
                    temp += 1
                else:
                    if img_list[l] not in jsonShapeError:
                        jsonShapeError[img_list[l]] = []
                    jsonShapeError[img_list[l]].append({"gt": int(class_number_truth[i]), "pred": int(shapes[i])})

                if colors[i] == colour_number_truth[i]:
                    acc_color += 1
                    temp_color += 1
                else:
                    if img_list[l] not in jsonColorError:
                        jsonColorError[img_list[l]] = []
                    if i % 2 == 0:
                        gt_probable = error_detector.probable_colors(int(colour_number_truth[i]),
                                                                     int(colour_number_truth[i + 1]))
                        jsonColorError[img_list[l]].append({"gt1": int(colour_number_truth[i]), "pred1": colors[i],
                                                            "gt2": int(colour_number_truth[i + 1]),
                                                            "pred2": colors[i + 1],
                                                            "shape": colors_pred[int(i / 2)],
                                                            "gt_probable": gt_probable})
                    else:
                        gt_probable = error_detector.probable_colors(int(colour_number_truth[i - 1]),
                                                                     int(colour_number_truth[i]))
                        jsonColorError[img_list[l]].append(
                            {"gt1": int(colour_number_truth[i - 1]), "pred1": colors[i - 1],
                             "gt2": int(colour_number_truth[i]), "pred2": colors[i],
                             "shape": colors_pred[int(i / 2)], "gt_probable": gt_probable})
            if write_image:
                for i in range(8):
                    if not os.path.isdir('version2/outer/'):
                        os.mkdir('version2/outer/')
                    cv2.imwrite('version2/outer/'+str(class_number_truth[i])+'__PRED-'+str(shapes[i])+'__'+str(i+1)+'_'+img_list[l], pred_list[i])

                for i in range(4):
                    if not os.path.isdir('version2/inner/'):
                        os.mkdir('version2/inner/')
                    cv2.imwrite('version2/inner/PRED-'+str(colors_pred[i])+'__'+str(i+1)+'_'+img_list[l], pred_list_inner[i])

        if print_flag:
            print("Accuracy (Shape): ", acc_class / perm_count)
            print("Accuracy (Color): ", acc_color / perm_count)

    print("Final Accuracy (Shape): ", acc_class / perm_count)
    print("Final Accuracy (Color): ", acc_color / perm_count)

    with open('shape_errors_v2.json', 'w') as json_file2:
        json.dump(jsonShapeError, json_file2, indent=2)
    with open('color_errors_v2.json', 'w') as json_file2:
        json.dump(jsonColorError, json_file2, indent=2)
'''

if __name__ == '__main__':
    upload()

