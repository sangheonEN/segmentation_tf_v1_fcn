import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import math
import cv2

class Batch_Create(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        print(idx)

        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        return batch_x, batch_y

def data_load(input_path, mask_path):
    # npy array load 후 patch function에 넣기
    image_list = os.listdir(input_path)
    image_list = [file for file in image_list if file.startswith('input_')]
    label_list = os.listdir(mask_path)
    label_list = [file for file in label_list if file.startswith('mask_')]

    iteration_num = len(image_list)

    return image_list, label_list, iteration_num

def read_image_normal(data_list, path):
    data_Num = len(data_list)
    total_image_list = []

    for i in range(data_Num):
        image_array = cv2.imread(os.path.join(path, data_list[i]), cv2.IMREAD_COLOR)
        # image_array = image_array / 255.0 -> 일단 하지말자 fcn_model에서는 전이 학습할때 이걸 normalization 해주는거 같음.
        total_image_list.append(image_array)

    return total_image_list

def read_mask_normal(data_list, path):
    data_Num = len(data_list)
    total_mask_list = []

    for i in range(data_Num):
        image_array = cv2.imread(os.path.join(path, data_list[i]), cv2.IMREAD_GRAYSCALE)
        # image_array = image_array / 255.0  -> mask image는 preprocess할때 normalization 완료.
        total_mask_list.append(image_array)

    return total_mask_list

def create_patch(npy_image, npy_mask, patch_size, overlay):
    # input : (batch_size, h, w, 3), mask : (batch_size, h, w, 1)

    step = patch_size - overlay
    for row in range(0, npy_image.shape[1] - overlay, step):
        for col in range(0, npy_image.shape[2] - overlay, step):
            patch_image_height = patch_size if npy_image.shape[1] - row > patch_size else npy_image.shape[1] - row
            patch_image_width = patch_size if npy_image.shape[2] - col > patch_size else npy_image.shape[1] - col

            patch_image = npy_image[:, row : row + patch_image_height, col : col + patch_image_width]
            patch_mask = npy_mask[:, row : row + patch_image_height, col : col + patch_image_width]

            # zero padding
            if patch_image_height < patch_size or patch_image_width < patch_size:
                pad_height = patch_size - patch_image_height
                pad_width = patch_size - patch_image_width
                patch_image = np.pad(patch_image, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)), 'constant')
                patch_mask = np.pad(patch_mask, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)), 'constant')

            yield patch_image, patch_mask, row, col

