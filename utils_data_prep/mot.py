# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE_MIT file in the project root for full license information.


# importing packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
from os import path
import cv2
import csv
import xml.etree.ElementTree as ET


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def crop_images(img_path='MOT16_04/img1', output_path = 'MOT16_04/cropped', prefix='', top=260, left=0,
                right=700, bottom=0, scale_factor=0.58):

    idx = 0
    for root_dir, dirs, files in os.walk(img_path):

        if len(files) > 0:
            for filename in files:
                idx+=1
                print(idx)
                filepath = os.path.join(root_dir, filename)
                orig = cv2.imread(filepath)
                img_name = filename.split('.')[0]
                crop_img = orig[top:orig.shape[0]-bottom, left:orig.shape[1]-right]
                width = int(crop_img.shape[1] * scale_factor)
                height = int(crop_img.shape[0] * scale_factor)
                scaled_img = cv2.resize(crop_img, (width, height))
                cv2.imwrite(os.path.join(output_path, prefix + img_name + '.jpg'), scaled_img)

    return

# crop_images(img_path='MOT20_05/img1', output_path = 'MOT20_05/cropped', prefix='', top=250, left=440,
#                 right=0, bottom=0, scale_factor=0.58)

# crop_images(img_path='MOT16_04/img1', output_path = 'MOT16_04/cropped', prefix='', top=0, left=0,
#                 right=1212, bottom=500, scale_factor=0.497)

# crop_images(img_path='MOT20_03/img1', output_path = 'MOT20_03/cropped', prefix='', top=0, left=0,
#                 right=523, bottom=348, scale_factor=0.541)


def get_bbs(dir_path='MOT20_05', prefix='mot20_05_', cropped=[440, 1654, 250, 1080], scale_factor=0.58):
    """
    Takes directory as input and populates xml files for all images. If the image is to be cropped, cropped is an input
    list that indicates bounding pixels for cropping image [left, right, top, bottom]. Then the image is scaled using
    the scale_factor.
    """


    print('populating bb dictionary')
    #look through annotations and store them in dictionary
    img_set = {}
    idx = 0

    with open(os.path.join(dir_path, 'gt', 'gt.txt'), 'r') as f:
        annot = [line.rstrip() for line in f]
        for obj in annot:
            idx+=1
            print(idx)
            params = obj.split(',')
            image_number = int(params[0])
            # initiate list for this image if not already in dictionary
            if image_number not in img_set:
                img_set[image_number] = []
            img_set[image_number].append(obj) # add bb info

    #generating xml file for each image
    print('populating xml files')
    img_sizes = {}
    idx = 0
    for root_dir, dirs, files in os.walk(os.path.join(dir_path, 'cropped')):

        if len(files) > 0:
            for filename in files:
                idx += 1
                if idx % 100 == 0:
                    print(idx)
                image_name = filename.split('.')[0]
                image_number = int(float(image_name))

                if image_name not in img_sizes:
                    filepath = os.path.join(root_dir, filename)
                    orig = cv2.imread(filepath)
                    img_sizes[image_name] = orig.shape

                tree = ET.parse(os.path.join(dir_path, 'template.xml'))
                root = tree.getroot()
                root.find('filename').text = prefix + image_name + '.jpg'
                root.find('source').find('image').text = image_name + '.jpg'
                root.find('size').find('width').text = str(img_sizes[image_name][1])
                root.find('size').find('height').text = str(img_sizes[image_name][0])

                for obj_ in img_set[image_number]:
                    param = obj_.split(',')
                    if param[7] == '1' or param[7] == '7': # only if object is pedestrian (static or walking)

                        # get bb parameters
                        xmin_v = int(scale_factor*clamp(float(param[2]) - cropped[0], 0, cropped[1] - cropped[0]))
                        ymin_v = int(scale_factor*clamp(float(param[3]) - cropped[2], 0, cropped[3] - cropped[2]))
                        xmax_v = int(scale_factor*clamp(float(param[2]) + float(param[4]) - cropped[0], 0,
                                                            cropped[1] - cropped[0]))
                        ymax_v = int(scale_factor*clamp(float(param[3]) + float(param[5]) - cropped[2], 0,
                                                            cropped[3] - cropped[2]))

                        if (xmin_v != xmax_v) and (ymin_v != ymax_v):  # only write object that are inside cropped image
                            obj = ET.SubElement(root, 'object')
                            name = ET.SubElement(obj, 'name')
                            name.text = 'pedestrian'
                            bndbox = ET.SubElement(obj, 'bndbox')
                            xmin = ET.SubElement(bndbox, 'xmin')
                            ymin = ET.SubElement(bndbox, 'ymin')
                            xmax = ET.SubElement(bndbox, 'xmax')
                            ymax = ET.SubElement(bndbox, 'ymax')

                            xmin.text = str(xmin_v)
                            ymin.text = str(ymin_v)
                            xmax.text = str(xmax_v)
                            ymax.text = str(ymax_v)

                tree.write(os.path.join(dir_path, 'Annotations', prefix + image_name + '.xml'))

    return

# get_bbs(dir_path='MOT20_05', prefix='mot20_05_', cropped=[440, 1654, 250, 1080], scale_factor=0.58)
# get_bbs(dir_path='MOT16_04', prefix='mot16_04_', cropped=[0, 708, 0, 580], scale_factor=0.497)
# get_bbs(dir_path='MOT20_03', prefix='mot20_03_', cropped=[0, 650, 0, 532], scale_factor=0.541)

def plot_boxes(img_path='MOT20_05/cropped', annot_path = 'MOT20_05/Annotations',
               output_dir='MOT20_05/bbs', prefix='mot20_05_'):
    """
    Takes directory as input and saves all images in bbs directory with boundary boxes
    """
    idx = 0
    count_ped = 0

    for root_dir, dirs, files in os.walk(img_path):

        if len(files) > 0:
            for filename in files:
                idx += 1
                print(idx)
                filepath = os.path.join(root_dir, filename)
                orig = cv2.imread(filepath)
                img_name = filename.split('.')[0]

                tree = ET.parse(os.path.join(annot_path, prefix + img_name + '.xml'))
                root = tree.getroot()

                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    name = obj.find('name')
                    if name.text == 'pedestrian':
                        count_ped += 1

                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.imwrite(os.path.join(output_dir, img_name + '.jpg'), orig)

    print('pedestrian:' + str(count_ped))

    return


# plot_boxes(img_path='MOT20_05/cropped', annot_path = 'MOT20_05/Annotations',
#                output_dir='MOT20_05/bbs', prefix='mot20_05_')

# plot_boxes(img_path='MOT16_04/cropped', annot_path = 'MOT16_04/Annotations',
#                output_dir='MOT16_04/bbs', prefix='mot16_04_')

# plot_boxes(img_path='MOT20_03/cropped', annot_path = 'MOT20_03/Annotations',
#                output_dir='MOT20_03/bbs', prefix='mot20_03_')

def add_prefix_images(img_path = 'MOT20_05/cropped', prefix='mot20_05_'):
    idx = 0
    count_ped = 0
    count_cyc = 0
    count_bus = 0
    img_count = 0

    for root_dir, dirs, files in os.walk(img_path):

        if len(files) > 0:
            for filename in files:
                idx += 1
                print(idx)
                old_filepath = os.path.join(root_dir, filename)
                new_filepath = os.path.join(root_dir, prefix + filename)

                os.rename(old_filepath, new_filepath)

    return


# add_prefix_images(img_path = 'MOT20_05/cropped', prefix='mot20_05_')
# add_prefix_images(img_path = 'MOT16_04/cropped', prefix='mot16_04_')
# add_prefix_images(img_path = 'MOT20_03/cropped', prefix='mot20_03_')


def populate_image_list(img_path = 'MOT20_05/cropped', annot_path = 'MOT20_05/Annotations',
                        output_dir = 'MOT20_05'):
    idx = 0
    img_list_A = [] # list containing all images
    img_list_B = [] # list containing only 10% of images
    img_list_C = []  # list containing only 1% of images
    count_A = 0
    count_B = 0
    count_C = 0

    for root_dir, dirs, files in os.walk(img_path):
        for filename in files:
            idx += 1
            print(idx)
            img_name = filename.split('.')[0]

            tree = ET.parse(os.path.join(annot_path, img_name + '.xml'))
            root = tree.getroot()

            obj_count = 0
            for obj in root.findall('object'):
                obj_count += 1

            # adding image to list A
            img_list_A.append(img_name)
            count_A += obj_count

            if idx % 10 == 0:
                img_list_B.append(img_name)
                count_B += obj_count

            if idx % 100 == 0:
                img_list_C.append(img_name)
                count_C += obj_count

    img_list_A = sorted(img_list_A, reverse=True)
    img_list_B = sorted(img_list_B, reverse=True)
    img_list_C = sorted(img_list_C, reverse=True)

    with open(os.path.join(output_dir, 'train_all.txt'), 'w') as f:
        while len(img_list_A) > 0:
            f.write(img_list_A.pop())
            f.write("\n")

    with open(os.path.join(output_dir, 'train_10pct.txt'), 'w') as f:
        while len(img_list_B) > 0:
            f.write(img_list_B.pop())
            f.write("\n")

    with open(os.path.join(output_dir, 'train_1pct.txt'), 'w') as f:
        while len(img_list_C) > 0:
            f.write(img_list_C.pop())
            f.write("\n")

    print('all' + str(count_A) + ' pedestrians')
    print('10%' + str(count_B) + ' pedestrians')
    print('1%' + str(count_C) + ' pedestrians')

    return


# populate_image_list(img_path = 'MOT20_05/cropped', annot_path = 'MOT20_05/Annotations',
#                         output_dir = 'MOT20_05')

# populate_image_list(img_path = 'MOT16_04/cropped', annot_path = 'MOT16_04/Annotations',
#                         output_dir = 'MOT16_04')
#
# populate_image_list(img_path = 'MOT20_03/cropped', annot_path = 'MOT20_03/Annotations',
#                         output_dir = 'MOT20_03')
