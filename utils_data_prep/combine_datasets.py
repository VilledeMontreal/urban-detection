# Copyright (c) Ville de Montreal. All rights reserved.
# Licensed under the MIT license.
# See LICENSE_MIT file in the project root for full license information.


# Importing packages
import os
import argparse
from distutils.dir_util import copy_tree
import shutil
from shutil import copyfile
import torchvision.transforms.functional as FT
import xml.etree.ElementTree as ET
from PIL import Image
import cv2


def pascal_to_yolo_annotations(label_dict, input_dir, output_dir):
    """
    A function that converts from a Pascal VOC format of annotation with xml files to a format compatible with yolo
    which uses text files. For each image, the Pascal VOC xml file includes the image size, and the xmin, ymin, xmax,
    ymax coordinates for all objects. Yolo txt files have one row for each object, with class
    number, x_center, y_center, width and height. Coordinates are normalized in yolo txt files, but not in pascal
    voc format. Class numbers are zero-indexed (start from 0).

    inputs
    ------
    label_dict : dictionary containing class labels as keys and digits as values
    input_dir : directory path to pascal voc annotations
    output_dir : directory path where yolo txt files shall be saved
    """


    for dir_root, dirs, files in os.walk(input_dir):
        if len(files) > 0:
            for filename in files:
                filepath = os.path.join(dir_root, filename)
                image_number = filename.split('.')[0]

                # parse xml tree
                tree = ET.parse(filepath)
                root = tree.getroot()

                # get image shape
                size = root.find('size')
                width = size.find('width').text
                height = size.find('height').text

                # for each object, write line in text file
                with open(os.path.join(output_dir, image_number + '.txt'), 'w') as yolo_file:
                    for object in root.findall('object'):
                        name = object.find('name').text
                        xmin = object.find('bndbox').find('xmin').text
                        ymin = object.find('bndbox').find('ymin').text
                        xmax = object.find('bndbox').find('xmax').text
                        ymax = object.find('bndbox').find('ymax').text

                        # converting coordinates to yolo format (i.e. box center coordinate with width and height) and
                        # normalize values between 0 and 1
                        xcenter = str(((float(xmax)+float(xmin))/2)/float(width))
                        ycenter = str(((float(ymax)+float(ymin))/2)/float(height))
                        obj_width = str((float(xmax)-float(xmin))/float(width))
                        obj_height = str((float(ymax)-float(ymin))/float(height))
                        yolo_file.write(str(label_dict[name]) + ' ' + xcenter + ' ' + ycenter +
                                        ' ' + obj_width + ' ' + obj_height)
                        yolo_file.write("\n")

    return


def create_yolo_directories(train_list, val_list, test_i_list, test_o_list, image_dir, label_dir, output_dir):
    """
    Populates an object detection dataset with the yolo folder architecture.

    inputs
    ------
    train_list : list of image numbers for training
    val_list : list of image numbers for validation set
    test_i_list : list of image numbers for in-domain test set
    test_o_list : list of image numbers for out-domain test set
    image_dir : path to image directory
    label_dir : path to yolo annotations
    output_dir : path to output yolo dataset
    """

    # make directories following desired yolo structure
    os.makedirs(os.path.join(output_dir, 'labels'))
    os.makedirs(os.path.join(output_dir, 'labels', 'train'))
    os.makedirs(os.path.join(output_dir, 'labels', 'val'))
    os.makedirs(os.path.join(output_dir, 'labels', 'test_i'))
    os.makedirs(os.path.join(output_dir, 'labels', 'test_o'))
    os.makedirs(os.path.join(output_dir, 'images'))
    os.makedirs(os.path.join(output_dir, 'images', 'train'))
    os.makedirs(os.path.join(output_dir, 'images', 'val'))
    os.makedirs(os.path.join(output_dir, 'images', 'test_i'))
    os.makedirs(os.path.join(output_dir, 'images', 'test_o'))

    # for each element in training list, add image and label in yolo dataset directory
    with open(train_list, 'r') as f:
        t_list = [line.rstrip() for line in f]
        for image_num in t_list:
            label_path = os.path.join(label_dir, str(image_num) + '.txt')
            # copy label to yolo train folder
            copyfile(label_path, os.path.join(output_dir, 'labels', 'train', str(image_num) + '.txt'))

            # when combining cgmu dataset with additional datasets, images do not all have the same extension
            if image_num[0].isdigit():
                image_path = os.path.join(image_dir, str(image_num) +'.jpeg')
                copyfile(image_path, os.path.join(output_dir, 'images', 'train', str(image_num) + '.jpeg'))
            else:
                image_path = os.path.join(image_dir, str(image_num) + '.jpg')
                copyfile(image_path, os.path.join(output_dir, 'images', 'train', str(image_num) + '.jpg'))


    # for each element in validation list, add image and label in yolo dataset directory
    with open(val_list, 'r') as f:
        v_list = [line.rstrip() for line in f]
        for image_num in v_list:
            label_path = os.path.join(label_dir, str(image_num) + '.txt')
            image_path = os.path.join(image_dir, str(image_num) + '.jpeg')

            # copy label to yolo validation folder
            copyfile(label_path, os.path.join(output_dir, 'labels', 'val', str(image_num) + '.txt'))

            # copy image to yolo validation folder
            copyfile(image_path, os.path.join(output_dir, 'images', 'val', str(image_num) + '.jpeg'))


    # for each element in in-domain test list, add image and label in yolo dataset directory
    with open(test_i_list, 'r') as f:
        ti_list = [line.rstrip() for line in f]
        for image_num in ti_list:
            label_path = os.path.join(label_dir, str(image_num) + '.txt')
            image_path = os.path.join(image_dir, str(image_num) + '.jpeg')

            # copy label to yolo in-domain test folder
            copyfile(label_path, os.path.join(output_dir, 'labels', 'test_i', str(image_num) + '.txt'))

            # copy image to yolo in-domain test folder
            copyfile(image_path, os.path.join(output_dir, 'images', 'test_i', str(image_num) + '.jpeg'))


    # for each element in out-domain test list, add image and label in yolo dataset directory
    with open(test_o_list, 'r') as f:
        teo_list = [line.rstrip() for line in f]
        for image_num in teo_list:
            label_path = os.path.join(label_dir, str(image_num) + '.txt')
            image_path = os.path.join(image_dir, str(image_num) + '.jpeg')

            # copy label to yolo out-domain test folder
            copyfile(label_path, os.path.join(output_dir, 'labels', 'test_o', str(image_num) + '.txt'))

            # copy image to yolo out-domain test folder
            copyfile(image_path, os.path.join(output_dir, 'images', 'test_o', str(image_num) + '.jpg'))

    return

def resize(image, dims=(300, 300)):
    """
    A function to resize images.
    Since YOLO uses percent/fractional coordinates, then bounding boxes do not need to be updated with resize process,
    we can retain them.

    inputs
    ------
    image : image, a PIL Image
    dims : dimensions of image to be returned

    output
    ------
    newresized image
    """
    # Resize image
    new_image = FT.resize(image, dims)

    return new_image

def resize_images_in_directory(dir_path, dims=(300,300)):
    """
    A function to resize all images in a directory.
    Since YOLO uses percent/fractional coordinates, then bounding boxes do not need to be updated with resize process,
    we can retain them.

    inputs
    ------
    dir_path : path to directory containing images
    dims : dimensions of image to be returned
    """

    for dir_root, dirs, files in os.walk(dir_path):
        if len(files) > 0:
            for filename in files:
                filepath = os.path.join(dir_root, filename)

                image = Image.open(filepath, mode='r')
                new_image = resize(image, dims)
                new_image.save(filepath)

    return


def plot_boxes(img_path='MIO_TCD_Localization/train', annot_path = 'MIO_TCD_Localization/Annotations',
               output_dir='MIO_TCD_Localization/bbs', prefix='miotcd'):
    """
    Takes image and annotation directories as input, will add bounding boxes ans save new images in another directory

    inputs
    ------
    img_path : directory path for images
    annot_path : directory path for annotations
    output_dir : directory path for output images with bounding boxes
    prefix : prefix to be added to saved images name
    """

    # initialization of statistic variables
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

                # parsing xml tree
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

                    # no distinction between class, only to verify position of bounding boxes
                    cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.imwrite(os.path.join(output_dir, img_name + '.jpg'), orig)

    print('pedestrian:' + str(count_ped))
    print('vehicles:' + str(count_v))
    print('cyclists:' + str(count_cyc))
    print('buses:' + str(count_bus))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combines CGMU dataset with additional datasets and convert to yolo")
    parser.add_argument('-d', '--datasets', nargs='+', type=str, required=True,
                        help="list of datasets to combine (options: cgmu, miotcd, mot1604, mot2003, mot2005, pets)")
    parser.add_argument('-y', '--yolo', action='store_true', default=False,
                        help="format to be converted to yolo format, otherwise pascal voc")
    parser.add_argument('-o', '--output', default='./combined', type=str,
                        help="output folder (will be created automatically if not exists)")
    parser.add_argument('-s', '--motsubsample', default=1, type=int,
                        help="subsampling percentage for mot datasets. Options: 1, 10, 100  Default: 1")
    parser.add_argument('-t', '--miotcdsubsample', default='all', type=str,
                        help="subsampling strategy for miotcd. Options: 'all' 'no_vehicle'  Default: all")
    parser.add_argument('-r', '--resize', nargs='*', default=[], type=int,
                        help="resize dimensions if resizing is desired (e.g. 320 320), Default: no resizing ")

    args = parser.parse_args()


    mot_subsample_dict = {1: 'train_1pct.txt', 10:'train_10pct.txt', 100:'train_all.txt'}

    #verify if output folder exists, delete and recreate it
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
        os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, 'Annotations'))
    os.makedirs(os.path.join(args.output, 'JPEGImages'))
    os.makedirs(os.path.join(args.output, 'ImageSets'))
    os.makedirs(os.path.join(args.output, 'ImageSets', 'Main'))

    # verify that selected datasets exist
    set1 = set(args.datasets)
    set2 = set(['cgmu', 'miotcd', 'mot1604', 'mot2003', 'mot2005', 'pets'])
    is_subset = set1.issubset(set2)
    if not is_subset:
        print('Selected datasets not accepted. List of available datasets: cgmu, miotcd, mot1604, mot2003, '
              'mot2005 and pets')
    else:
        # copy all files from first dataset into output folder
        print('Combining datasets...')
        for ds in args.datasets:
            ds_tocopy = ds + '_voc'

            # combine images
            copy_tree(os.path.join(ds_tocopy, 'JPEGImages'), os.path.join(args.output, 'JPEGImages'))

            # combine annotations
            copy_tree(os.path.join(ds_tocopy, 'Annotations'), os.path.join(args.output, 'Annotations'))

            # combine image lists (imagesets)
            for root, dirs, files in os.walk(os.path.join(ds_tocopy, 'ImageSets', 'Main')):
                for file in files:
                    dest_path = os.path.join(args.output, 'ImageSets', 'Main')
                    src_path = os.path.join(ds_tocopy, 'ImageSets', 'Main')

                    dest_file = file
                    # if mot dataset, only copy lists that belong to subsampling strategy
                    if ds[0:3] == 'mot':
                        if mot_subsample_dict[args.motsubsample] != file:
                            continue
                        else:
                            dest_file = file.split('_')[0] + '.txt' #destination file is train.txt, not train_1pct.txt
                    elif ds[0:3] == 'mio':
                        if ('train_' + args.miotcdsubsample + '.txt') != file:
                            continue
                        else:
                            dest_file = file.split('_')[0] + '.txt' #destination file is train.txt, not train_all.txt


                    # if file does not exist in destination folder, simply copy it there
                    if not os.path.exists(os.path.join(dest_path, dest_file)):
                        shutil.copy(os.path.join(src_path, file), dest_path)
                        if dest_file != file: # if train_1pct.txt is copied, we rename it to train.txt
                            os.rename(os.path.join(dest_path, file), os.path.join(dest_path, dest_file))
                    else: # otherwise, merge txt files together
                        with open(os.path.join(dest_path, dest_file), 'r') as fp:
                            data = fp.read()
                        with open(os.path.join(src_path, file), 'r') as fp:
                            data2 = fp.read()
                        # merging
                        data += data2
                        # overwriting destination file
                        with open(os.path.join(dest_path, dest_file), 'w') as fp:
                            fp.write(data)

    # resizing images
    if args.resize:
        print('Resizing images to ' + str(args.resize[0]) + 'x' + str(args.resize[1]) + ' ...')
        resize_images_in_directory(os.path.join(args.output, 'JPEGImages'), dims=tuple(args.resize))

        if not args.yolo:
            print('Adjusting bounding box coordinates in Annotations files ...')
            # resizing requires that we change xmin, xmax, ymin and ymax for pascal voc format. For yolo, it is not
            # necessary as bb dimensions are expressed in ratios.
            for root, dirs, files in os.walk(os.path.join(args.output, 'Annotations')):
                for filename in files:
                    filepath = os.path.join(args.output, 'Annotations', filename)
                    image_number = filename.split('.')[0]

                    # parse xml tree
                    tree = ET.parse(filepath)
                    root = tree.getroot()

                    # get image shape
                    size = root.find('size')
                    width = size.find('width').text
                    size.find('width').text = str(args.resize[1])
                    x_ratio = float(args.resize[1]) / float(width)
                    height = size.find('height').text
                    size.find('height').text = str(args.resize[0])
                    y_ratio = float(args.resize[0]) / float(height)

                    # for each object, update xmin, ymin, xmax and ymax according to resize factor
                    for object in root.findall('object'):
                        name = object.find('name').text
                        xmin = object.find('bndbox').find('xmin').text
                        ymin = object.find('bndbox').find('ymin').text
                        xmax = object.find('bndbox').find('xmax').text
                        ymax = object.find('bndbox').find('ymax').text
                        object.find('bndbox').find('xmin').text = str(float(xmin) * x_ratio)
                        object.find('bndbox').find('ymin').text = str(float(ymin) * y_ratio)
                        object.find('bndbox').find('xmax').text = str(float(xmax) * x_ratio)
                        object.find('bndbox').find('ymax').text = str(float(ymax) * y_ratio)

                    tree.write(filepath)


            ## The block below can be uncommented for debugging purposes to verify bounding boxes
            # plot_boxes(img_path=os.path.join(args.output, 'JPEGImages'),
            #            annot_path=os.path.join(args.output, 'Annotations'),
            #                output_dir=os.path.join('./', 'bbs'), prefix='')


    # if desired format is yolo, convert from PascalVOC format to yolo format
    if args.yolo:
        print('Converting format to yolo format...')

        #create directory to temporarily store labels
        os.makedirs(os.path.join(args.output, 'temp_labels'))

        #convert annotations from pascal to yolo format
        label_dict = {'vehicle': 0, 'pedestrian': 1, 'construction': 2, 'bus': 3, 'cyclist': 4}
        pascal_to_yolo_annotations(label_dict=label_dict,
                                   input_dir=os.path.join(args.output, 'Annotations'),
                                   output_dir=os.path.join(args.output, 'temp_labels'))

        create_yolo_directories(train_list=os.path.join(dest_path, 'train.txt'),
                                val_list=os.path.join(dest_path, 'val.txt'),
                                test_i_list=os.path.join(dest_path, 'test_i.txt'),
                                test_o_list=os.path.join(dest_path, 'test_o.txt'),
                                image_dir=os.path.join(args.output, 'JPEGImages'),
                                label_dir=os.path.join(args.output, 'temp_labels'),
                                output_dir=args.output)

        # delete old pascal directories, no longer needed
        shutil.rmtree(os.path.join(args.output, 'JPEGImages'))
        shutil.rmtree(os.path.join(args.output, 'temp_labels'))
        shutil.rmtree(os.path.join(args.output, 'Annotations'))
        shutil.rmtree(os.path.join(args.output, 'ImageSets'))
