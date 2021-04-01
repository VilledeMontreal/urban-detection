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
import xml.etree.ElementTree as ET


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def crop_images(img_path='extra_voc/JPEGImages', output_path = 'extra_voc/cropped', prefix='pets', top=102, left=40,
                right=40, bottom=0):

    for root_dir, dirs, files in os.walk(img_path):

        if len(files) > 0:
            for filename in files:
                filepath = os.path.join(root_dir, filename)
                orig = cv2.imread(filepath)
                img_name = filename.split('.')[0]
                crop_img = orig[top:orig.shape[0]-bottom, left:orig.shape[1]-right]

                cv2.imwrite(os.path.join(output_path, prefix + img_name + '.jpg'), crop_img)

    return

# crop_images(img_path='extra_voc/JPEGImages', output_path = 'extra_voc/cropped', prefix='pets', top=102, left=40,
#                 right=40, bottom=0)
#


def get_bbs(dir_path='extra_voc', prefix='', cropped=[40, 728, 102, 576]):
    """
    Takes directory as input and populates xml files for all images. If the image is to be cropped, cropped is an input
    list that indicates bounding pixels for cropping image
    """

    with open(os.path.join(dir_path, 'gt.txt'), 'r') as f:
        annot = [line.rstrip() for line in f]

        current_img = -1
        for obj in annot:
            param = obj.split(',')
            img = int(param[0])
            image_name = '00' + '{:04d}'.format(img)

            # if object belongs to same image, continue building xml
            if img == current_img:

                # create new object
                xmin_v = str(clamp(float(param[2]) - cropped[0], 0, cropped[1] - cropped[0]))
                ymin_v = str(clamp(float(param[3]) - cropped[2], 0, cropped[3] - cropped[2]))
                xmax_v = str(clamp(float(param[2]) + float(param[4]) - cropped[0], 0, cropped[1] - cropped[0]))
                ymax_v = str(clamp(float(param[3]) + float(param[5]) - cropped[2], 0, cropped[3] - cropped[2]))

                if (xmin_v != xmax_v) and (ymin_v != ymax_v): # only write object that are inside cropped image

                    obj = ET.SubElement(root, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'pedestrian'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin = ET.SubElement(bndbox, 'xmin')
                    ymin = ET.SubElement(bndbox, 'ymin')
                    xmax = ET.SubElement(bndbox, 'xmax')
                    ymax = ET.SubElement(bndbox, 'ymax')

                    xmin.text = xmin_v
                    ymin.text = ymin_v
                    xmax.text = xmax_v
                    ymax.text = ymax_v

                    tree.write(os.path.join(dir_path, 'Annotations', prefix + image_name + '.xml'))

            else: # otherwise create new xml
                tree = ET.parse(os.path.join(dir_path, 'template.xml'))
                root = tree.getroot()
                root.find('filename').text = prefix + image_name + '.jpg'
                root.find('source').find('image').text = image_name + '.jpg'
                root.find('size').find('width').text = str(cropped[1] - cropped[0])
                root.find('size').find('height').text = str(cropped[3] - cropped[2])

                # create new object
                xmin_v = str(clamp(float(param[2]) - cropped[0], 0, cropped[1] - cropped[0]))
                ymin_v = str(clamp(float(param[3]) - cropped[2], 0, cropped[3] - cropped[2]))
                xmax_v = str(clamp(float(param[2]) + float(param[4]) - cropped[0], 0, cropped[1] - cropped[0]))
                ymax_v = str(clamp(float(param[3]) + float(param[5]) - cropped[2], 0, cropped[3] - cropped[2]))

                if (xmin_v != xmax_v) and (ymin_v != ymax_v):  # only write object that are inside cropped image

                    obj = ET.SubElement(root, 'object')
                    name = ET.SubElement(obj, 'name')
                    name.text = 'pedestrian'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    xmin = ET.SubElement(bndbox, 'xmin')
                    ymin = ET.SubElement(bndbox, 'ymin')
                    xmax = ET.SubElement(bndbox, 'xmax')
                    ymax = ET.SubElement(bndbox, 'ymax')

                    xmin.text = xmin_v
                    ymin.text = ymin_v
                    xmax.text = xmax_v
                    ymax.text = ymax_v

                    tree.write(os.path.join(dir_path, 'Annotations', prefix + image_name + '.xml'))

            current_img = img

    return

# get_bbs('extra_voc', 'pets')

def add_cars(annot_path='extra_voc/Annotations'): #not required if we crop images
    """
    take annotation directory path as input and add two vehicle objects in pets images
    """
    idx = 0

    for root_dir, dirs, files in os.walk(annot_path):

        if len(files) > 0:
            for filename in files:
                tree = ET.parse(os.path.join(annot_path, filename))
                root = tree.getroot()

                # create vehicle object 1
                obj = ET.SubElement(root, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = 'vehicle'
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')
                xmin.text = '652'
                ymin.text = '99'
                xmax.text = '724'
                ymax.text = '43'

                # create vehicle object 2
                obj = ET.SubElement(root, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = 'vehicle'
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                ymin = ET.SubElement(bndbox, 'ymin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymax = ET.SubElement(bndbox, 'ymax')
                xmin.text = '733'
                ymin.text = '93'
                xmax.text = '768'
                ymax.text = '46'

                tree.write(os.path.join(annot_path, filename))

#add_cars(annot_path='extra_voc/Annotations')

def plot_boxes(img_path='extra_voc/JPEGImages', annot_path = 'extra_voc/Annotations', output_dir='extra_voc/bbs', prefix='pets'):
    """
    Takes directory as input and saves all images in bbs directory with boundary boxes
    """
    idx = 0
    count_ped = 0
    count_v = 0

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
                    else:
                        count_v += 1
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.imwrite(os.path.join(output_dir, img_name + '.jpg'), orig)

    print('pedestrian:' + str(count_ped))
    print('vehicles:' + str(count_v))

    return

# plot_boxes(img_path='extra_voc/cropped', annot_path = 'extra_voc/Annotations', output_dir='extra_voc/bbs', prefix='')

def populate_image_list(img_path = 'extra_voc/cropped', output_file = 'extra_voc/ImageSets/Main/train.txt'):
    idx = 0
    img_list = []
    for root_dir, dirs, files in os.walk(img_path):
        for filename in files:
            idx += 1
            print(idx)
            img_list.append(filename.split('.')[0])

    img_list = sorted(img_list, reverse=True)

    with open(output_file, 'w') as f:
        while len(img_list) > 0:
            f.write(img_list.pop())
            f.write("\n")
    return


# populate_image_list(img_path = 'extra_voc/cropped', output_file = 'extra_voc/ImageSets/Main/train.txt')
