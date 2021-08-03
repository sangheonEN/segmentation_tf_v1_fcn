from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import datetime
from six.moves import xrange
import gc
import os
import cv2
import data_load as DATA_LOAD
import util as ut
import model as m

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_integer("Epoch", "20", "Total Epoch")
tf.flags.DEFINE_integer("patch_size", "768", "patch size")
tf.flags.DEFINE_integer("overlay", "256", "overlay")
tf.flags.DEFINE_string("logs_dir", "log", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "datasets", "path to data directory")
tf.flags.DEFINE_string("gpu_idx", "0", "gpu_idx = 0")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test")
tf.flags.DEFINE_string('optimal_model', "model.ckpt-15", "ckpt optimal model")

train_input_path = FLAGS.data_dir + '/raw_data/train/input'
train_mask_path = FLAGS.data_dir + '/raw_data/train/mask'

anno_train_path = FLAGS.data_dir + '/raw_data/train/aug_mask'
input_train_path = FLAGS.data_dir + '/raw_data/train/aug_input'

def anno_visualization(valid_anno_path, input_image, mask, count_m):

    # epoch 마다 valid image 저장 folder 생성
    valid_anno_mask_dir_save_epoch = os.path.join(valid_anno_path, 'anno_mask')
    valid_anno_input_dir_save_epoch = os.path.join(valid_anno_path, 'anno_input')
    if not os.path.exists(valid_anno_mask_dir_save_epoch):
        os.makedirs(valid_anno_mask_dir_save_epoch)
    if not os.path.exists(valid_anno_input_dir_save_epoch):
        os.makedirs(valid_anno_input_dir_save_epoch)

    # image pred 저장.
    for i in range(input_image.shape[0]):
        input_mask = input_image[i]
        cv2.imwrite(valid_anno_input_dir_save_epoch + '/annotation_input_%03d.png' %count_m, input_mask)
        anno_mask = mask[i] * 255
        cv2.imwrite(valid_anno_mask_dir_save_epoch + '/annotation_mask_%03d.png' %count_m, anno_mask)
        count_m += 1


def data_load(input_path, mask_path):
    # npy array load 후 patch function에 넣기
    image_list = os.listdir(input_path)
    image_list = [file for file in image_list if file.endswith('.png')]
    label_list = os.listdir(mask_path)
    label_list = [file for file in label_list if file.endswith('.png')]

    iteration_num = len(image_list)

    return image_list, label_list, iteration_num


def data_augmentation_flip(img):
    try:
        flipped_img = np.fliplr(img)

        return flipped_img

    except Exception as e:
        print(str(e))

def data_augmentation_flip_rotation(flipped_img, rows, cols):
    try:
        N1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        N2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        N3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)

        dst4 = cv2.warpAffine(flipped_img, N1, (cols, rows))
        dst5 = cv2.warpAffine(flipped_img, N2, (cols, rows))
        dst6 = cv2.warpAffine(flipped_img, N3, (cols, rows))

        return dst4, dst5, dst6

    except Exception as e:
        print(str(e))

def data_augmentation_rotation(img, rows, cols):
    try:
        M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        M3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)

        dst1 = cv2.warpAffine(img, M1, (cols, rows))
        dst2 = cv2.warpAffine(img, M2, (cols, rows))
        dst3 = cv2.warpAffine(img, M3, (cols, rows))

        return dst1, dst2, dst3

    except Exception as e:
        print(str(e))

def main(argv=None):

    # data load
    # data_load (train), read image, read mask
    train_x, train_y, train_iter = data_load(train_input_path, train_mask_path)

    train_images_list = DATA_LOAD.read_image_normal(train_x, train_input_path)
    train_masks_list = DATA_LOAD.read_mask_normal(train_y, train_mask_path)

    batch_array = DATA_LOAD.Batch_Create(train_images_list, train_masks_list, FLAGS.batch_size)

    # train_images, masks array가 계속 memory를 가지고 있음. 그래서 generator 생성 후 del.
    del train_images_list
    del train_masks_list
    gc.collect()

    count_input = 0
    count_mask = 0

    # batch create
    for x_list, y_list in batch_array:

        batch_x = np.array(x_list)
        batch_y = np.array(y_list)

        image_rotation = np.squeeze(batch_x, axis=0)
        mask_rotation = np.squeeze(batch_y, axis=0)
        mask_rotation = np.expand_dims(mask_rotation, axis=-1)
        rows, cols, img_d = image_rotation.shape[:]
        rows_m, cols_m, img_d = mask_rotation.shape[:]

        # 원본 저장.
        savename_img = input_train_path + '/input/valid_input_%07d.png'% count_input
        count_input += 1
        savename_mask = anno_train_path + '/mask/valid_mask_%07d.png'% count_mask
        count_mask += 1

        cv2.imwrite(savename_img, image_rotation)
        cv2.imwrite(savename_mask, mask_rotation)

        # rotation 저장
        img_rotation_90, img_rotation_180, img_rotation_270 = data_augmentation_rotation(image_rotation, rows, cols)
        mask_rotation_90, mask_rotation_180, mask_rotation_270 = data_augmentation_rotation(mask_rotation, rows_m, cols_m)

        savename_img_rotation_90 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1
        savename_img_rotation_180 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1
        savename_img_rotation_270 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1

        savename_mask_rotation_90 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1
        savename_mask_rotation_180 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1
        savename_mask_rotation_270 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1

        cv2.imwrite(savename_img_rotation_90, img_rotation_90)
        cv2.imwrite(savename_img_rotation_180, img_rotation_180)
        cv2.imwrite(savename_img_rotation_270, img_rotation_270)

        cv2.imwrite(savename_mask_rotation_90, mask_rotation_90)
        cv2.imwrite(savename_mask_rotation_180, mask_rotation_180)
        cv2.imwrite(savename_mask_rotation_270, mask_rotation_270)

        # flip 저장
        img_flip = data_augmentation_flip(image_rotation)
        mask_flip = data_augmentation_flip(mask_rotation)

        img_flip_rotation_90, img_flip_rotation_180, img_flip_rotation_270 = data_augmentation_flip_rotation(img_flip, rows, cols)
        mask_flip_rotation_90, mask_flip_rotation_180, mask_flip_rotation_270 = data_augmentation_flip_rotation(mask_flip, rows_m, cols_m)

        savename_img_flip = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1
        savename_img_flip_rotation_90 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1
        savename_img_flip_rotation_180 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1
        savename_img_flip_rotation_270 = input_train_path + '/train_input_%07d.png'% count_input
        count_input += 1

        savename_mask_flip = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1
        savename_mask_flip_rotation_90 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1
        savename_mask_flip_rotation_180 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1
        savename_mask_flip_rotation_270 = anno_train_path + '/train_mask_%07d.png'% count_mask
        count_mask += 1

        cv2.imwrite(savename_img_flip, img_flip)
        cv2.imwrite(savename_img_flip_rotation_90, img_flip_rotation_90)
        cv2.imwrite(savename_img_flip_rotation_180, img_flip_rotation_180)
        cv2.imwrite(savename_img_flip_rotation_270, img_flip_rotation_270)

        cv2.imwrite(savename_mask_flip, mask_flip)
        cv2.imwrite(savename_mask_flip_rotation_90, mask_flip_rotation_90)
        cv2.imwrite(savename_mask_flip_rotation_180, mask_flip_rotation_180)
        cv2.imwrite(savename_mask_flip_rotation_270, mask_flip_rotation_270)



if __name__ == "__main__":
    tf.app.run()



