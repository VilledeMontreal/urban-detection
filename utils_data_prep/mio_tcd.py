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


def keep_only_desired_objects(workdir = 'MIO_TCD_Localization', labels_to_keep = ['bus', 'bicycle', 'pedestrian']):
    """
    This function will delete all images that do not contain at least one of the label to keep. This is done so as to
    increase the number of objects within a list of under-represented labels.
    """

    with open(os.path.join(workdir, 'gt_train.csv'), 'r') as f:
        annot = [line.rstrip() for line in f]

        img_to_keep = set()

        for annotation in annot:
            params = annotation.split(',')
            if params[1] in labels_to_keep:
                img_to_keep.add(params[0])

    idx = 0
    keep_count = 0
    remove_count = 0
    for root_dir, dirs, files in os.walk(os.path.join(workdir, 'train')):

        if len(files) > 0:
            for filename in files:
                idx += 1
                print(idx)
                filepath = os.path.join(root_dir, filename)
                img_name = filename.split('.')[0]
                if img_name in img_to_keep:
                    keep_count += 1
                else:
                    remove_count += 1
                    if os.path.exists(filepath):
                        os.remove(filepath)

            print(str(keep_count) + ' images kept')
            print(str(remove_count) + ' images removed')

#keep_only_desired_objects(workdir = 'MIO_TCD_Localization', labels_to_keep = ['bus', 'bicycle', 'pedestrian'])



def get_bbs(dir_path='MIO_TCD_Localization', prefix='miotcd'):
    """
    Takes directory as input and populates xml files for all images.
    """

    #populate set of images in directory
    print('populating set of images and dict of image sizes')
    img_set = set()
    img_sizes = {}
    idx = 0
    for root_dir, dirs, files in os.walk(os.path.join(dir_path, 'train')):

        if len(files) > 0:
            for filename in files:
                idx += 1
                if idx % 100 == 0:
                    print(idx)
                img_name = filename.split('.')[0]
                img_set.add(img_name)

                if img_name not in img_sizes:
                    filepath = os.path.join(root_dir, filename)
                    orig = cv2.imread(filepath)
                    img_sizes[img_name] = orig.shape


    print('populating xml files')
    #look through annotations and only keep ones related to kept images
    idx = 0
    with open(os.path.join(dir_path, 'gt_train.csv'), 'r') as f:
        annot = [line.rstrip() for line in f]

        current_img = '-1'
        for obj in annot:
            param = obj.split(',')
            image_name = param[0]
            idx +=1
            if idx % 100 == 0 :
                print(idx)

            if image_name in img_set:
                if image_name == current_img:

                    # create new object
                    xmin_v = param[2]
                    ymin_v = param[3]
                    xmax_v = param[4]
                    ymax_v = param[5]

                    obj = ET.SubElement(root, 'object')
                    name = ET.SubElement(obj, 'name')

                    if param[1] == 'bicycle':
                        name.text = 'cyclist'
                    elif param[1] == 'pedestrian':
                        name.text = 'pedestrian'
                    elif param[1] == 'bus':
                        name.text = 'bus'
                    else:
                        name.text = 'vehicle'

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


                else: # otherwise create new xml
                    tree = ET.parse(os.path.join(dir_path, 'template.xml'))
                    root = tree.getroot()
                    root.find('filename').text = prefix + image_name + '.jpg'
                    root.find('source').find('image').text = image_name + '.jpg'
                    root.find('size').find('width').text = str(img_sizes[image_name][1])
                    root.find('size').find('height').text = str(img_sizes[image_name][0])

                    # create new object
                    xmin_v = param[2]
                    ymin_v = param[3]
                    xmax_v = param[4]
                    ymax_v = param[5]


                    obj = ET.SubElement(root, 'object')
                    name = ET.SubElement(obj, 'name')

                    if param[1] == 'bicycle':
                        name.text = 'cyclist'
                    elif param[1] == 'pedestrian':
                        name.text = 'pedestrian'
                    elif param[1] == 'bus':
                        name.text = 'bus'
                    else:
                        name.text = 'vehicle'

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

                current_img = image_name

    return

# get_bbs('MIO_TCD_Localization', 'miotcd')


def plot_boxes(img_path='MIO_TCD_Localization/train', annot_path = 'MIO_TCD_Localization/Annotations',
               output_dir='MIO_TCD_Localization/bbs', prefix='miotcd'):
    """
    Takes directory as input and saves all images in bbs directory with boundary boxes
    """
    idx = 0
    count_ped = 0
    count_cyc = 0
    count_bus = 0
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
                    elif name.text == 'bus':
                        count_bus += 1
                    elif name.text == 'cyclist':
                        count_cyc += 1
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
    print('cyclists:' + str(count_cyc))
    print('buses:' + str(count_bus))

    return


# plot_boxes(img_path='MIO_TCD_Localization/train', annot_path = 'MIO_TCD_Localization/Annotations',
#                output_dir='MIO_TCD_Localization/bbs', prefix='miotcd')


def count_classes_without_vehicles(img_path='MIO_TCD_Localization/train',
                                   annot_path = 'MIO_TCD_Localization/Annotations', prefix='miotcd'):
    """
    Goes through all images and look at objects in each, counts the number of each class, while disregarding all
    images that have vehicles.
    """
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
                filepath = os.path.join(root_dir, filename)
                img_name = filename.split('.')[0]

                img_c_ped = 0
                img_c_cyc = 0
                img_c_bus = 0
                img_c_v = 0

                tree = ET.parse(os.path.join(annot_path, prefix + img_name + '.xml'))
                root = tree.getroot()

                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name.text == 'pedestrian':
                        img_c_ped += 1
                    elif name.text == 'bus':
                        img_c_bus += 1
                    elif name.text == 'cyclist':
                        img_c_cyc += 1
                    else:
                        img_c_v += 1

                if img_c_v == 0:
                    count_ped += img_c_ped
                    count_cyc += img_c_cyc
                    count_bus += img_c_bus
                    img_count += 1

    print('pedestrian:' + str(count_ped))
    print('cyclists:' + str(count_cyc))
    print('buses:' + str(count_bus))
    print('images total:' + str(img_count))

    return


# count_classes_without_vehicles(img_path='MIO_TCD_Localization/train',
#                                    annot_path = 'MIO_TCD_Localization/Annotations', prefix='miotcd')


def add_prefix_images(img_path = 'MIO_TCD_Localization/train', annot_path = 'MIO_TCD_Localization/Annotations'):
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


# add_prefix_images(img_path = 'MIO_TCD_Localization/train', prefix='miotcd')


def populate_image_list(img_path = 'MIO_TCD_Localization/train', annot_path = 'MIO_TCD_Localization/Annotations',
                        output_dir = 'MIO_TCD_Localization'):
    idx = 0
    img_list_A = [] # list containing all images
    img_list_B = [] # list containing only image without vehicles

    for root_dir, dirs, files in os.walk(img_path):
        for filename in files:
            idx += 1
            print(idx)
            img_name = filename.split('.')[0]

            # adding image to list A
            img_list_A.append(img_name)

            img_c_v = 0

            tree = ET.parse(os.path.join(annot_path, img_name + '.xml'))
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name')
                if name.text == 'vehicle':
                    img_c_v += 1

            if img_c_v == 0:
                # adding image to list B
                img_list_B.append(img_name)

    img_list_A = sorted(img_list_A, reverse=True)
    img_list_B = sorted(img_list_B, reverse=True)

    with open(os.path.join(output_dir, 'train_all.txt'), 'w') as f:
        while len(img_list_A) > 0:
            f.write(img_list_A.pop())
            f.write("\n")

    with open(os.path.join(output_dir, 'train_no_vehicle.txt'), 'w') as f:
        while len(img_list_B) > 0:
            f.write(img_list_B.pop())
            f.write("\n")
    return


# populate_image_list(img_path = 'MIO_TCD_Localization/train', output_dir = 'MIO_TCD_Localization')
