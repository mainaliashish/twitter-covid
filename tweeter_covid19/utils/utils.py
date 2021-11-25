"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/14/2019
"""

import os
import re

import chardet
import numpy as np
import pandas as pd
import uuid


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except NotADirectoryError:
            pass


def list_files(directory, extensions=None, shuffle=False):
    """
    Lists files in a directory
    :return: list -> gives all files path
    """

    filenames = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)

            if extensions is not None:
                if file_path.endswith(tuple(extensions)):
                    filenames.append(file_path)
            else:
                filenames.append(file_path)
    if shuffle:
        np.random.shuffle(filenames)
    return filenames


def read_from_csv(path, column_name):
    df = pd.read_csv(path)
    return list(df[column_name])


def get_all_directory(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for filename in dirs:
            files_list.append(filename)
        return files_list


def get_all_files(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            files_list.append(filename)
        return files_list


def check_directory(path):
    if os.path.exists(path):
        return True
    else:
        print(path, " doesn't exist!")
        return False


def read_data_from_file(path=None, encoding='utf-8'):
    assert path is not None
    fp = open(path, "r", encoding=encoding, errors='ignore')
    path = [line.strip() for line in fp]
    fp.close()
    return path


def write_text_file(path=None, data=None):
    if path is None or data is None:
        return None
    fp = open(path, 'w', encoding='utf-8')
    fp.write(data)
    fp.close()


img_ext = ['.jpg', '.jpeg', '.bmp', '.png', '.JPG', '.JPEG', '.ppm', '.pgm', '.webp']
vid_ext = ['.mov', '.flv', '.mpeg', '.mpg', '.mp4', '.mvk', '.avi', '.3gp', '.webm']


def bbox_wh_to_ltrb(bbox):
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    return bbox


def bbox_ltrb_to_wh(bbox):
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return bbox


def get_uuid():
    return str(uuid.uuid4()).replace('-', '')


def get_int_uuid():
    return uuid.uuid4().fields[-1]


def read_file(filename):
    """
    :param filename:
    :return: list of lines in the given file
    """
    # return np.array([line.rstrip('\n') for line in open(filename, encoding='utf-8')])
    return [line.rstrip('\n') for line in open(filename, encoding='utf-8')]


def list_images(source, shuffle=False):
    """
    Generic function to list images from directory or file
    :return:  list of images in a given directory or file
    """
    files = []
    if os.path.isfile(source):
        if source.endswith('.txt'):
            files = read_file(source)
        elif source.endswith(tuple(img_ext)):
            files.append(source)
        if shuffle:
            np.random.shuffle(files)
    elif os.path.isdir(source):
        files = list_files(source, extensions=img_ext, shuffle=shuffle)

    return files


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def normalize_bbox(width, height, bbox):
    """
    :param width:
    :param height:
    :param bbox: [t, l ,  b, r]
    :return:
    """
    temp = bbox.copy()
    temp = np.divide(temp, np.array([height, width, height, width]).astype(np.float))
    return temp


def normalize_point(width, height, point):
    # x, y =  point

    if not isinstance(point, np.ndarray):
        return None
    temp = point.copy()
    temp = np.divide(temp, [height, width])
    return temp


def check_inside(point, box):
    if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]:
        return True
    return False


def sort_rois(rois, axis=1):
    """
    Sort rois in increasing order: y-axis
    :param rois:
    :return:
    """

    roi_ = []
    for roi in rois:
        if axis == 1:
            roi_.append(roi[1])
        elif axis == 0:
            roi_.append(roi[0])
        else:
            return rois

    try:
        rois = [x for _, x in sorted(zip(roi_, rois))]
    except ValueError as e:
        return rois
    return rois


def format_layout(page_layout):
    blocks = page_layout.blocks
    word_dict = {}
    bboxes = []
    maxx = 0
    for block in blocks:
        words = block.words
        for word in words:
            list1a = tuple(word.bbox)
            word_dict[list1a] = word.text.split("\n")[0]
            bboxes.append(word.bbox)
            x = int(word.bbox[2])
            if maxx < x:
                maxx = x
    grouped_bboxes = group_bbox(bboxes, margin=2)
    for id in grouped_bboxes.keys():
        grouped_bboxes[id] = sort_rois(grouped_bboxes[id], axis=0)
    formatted_text = ""
    for id, bboxes in grouped_bboxes.items():
        temp = 0
        for bbox in bboxes:
            x_dif = int(int(bbox[0] - temp) / maxx * 100)
            for _ in range(0, x_dif):
                formatted_text += " "
            formatted_text += word_dict[tuple(bbox)]
            temp = bbox[0]
        formatted_text += "\n"
    return formatted_text


def group_bbox(bboxes, margin=0):
    bboxes = sort_rois(bboxes, axis=1)
    grouped_bbox = {}
    keys = []
    key_ = 0
    for bbox in bboxes:
        y = int(bbox[1])
        if len(grouped_bbox) == 0:
            grouped_bbox[key_] = [bbox]
            new_keys = [y + i for i in range(-margin, margin + 1)]
            keys.extend(new_keys)
            continue
        if y in keys:
            grouped_bbox[key_].append(bbox)
        else:
            key_ += 1
            grouped_bbox[key_] = [bbox]
            new_keys = [y + i for i in range(-margin, margin + 1)]
            keys.extend(new_keys)
    return grouped_bbox


def get_file_name(name: object, _index: object, directory_only: object = False) -> object:
    """
    This function split the path of the file. eg.
    # print(get_file_name(files[0], 2)) gives target class
    # print(get_file_name(files[0], 1)) gives filename
    :param directory_only:
    :param name: path to split
    :param _index: index from the last
    :return:
    """
    _path = re.split("/|\\\\|//", name)
    if directory_only:
        return os.path.join(*_path[0:len(_path) - 1])
    return _path[len(_path) - _index]


def split_word_into_letters(word):
    return [char for char in word]


def write_text_file_linewise(file_path: object = None, data: object = None, encoding: object = None) -> object:
    assert type(data) == list
    if file_path is None or data is None:
        return None
    if encoding is None:
        file = open(file_path, 'w+')
    else:
        file = open(file_path, 'w+', encoding=encoding)
    for value in data:
        file.write(value + '\n')
    file.close()


def detect_encoding(file):
    raw_data = open(file, 'rb').read()
    result = chardet.detect(raw_data)
    return result['encoding']
