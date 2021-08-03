import os
import numpy as np
import cv2
import math

input_path = './datasets/raw_data/input/images'
input_rename_path = './datasets/raw_data/input/images_rename'
path_mask = './datasets/raw_data/label/mask'
path_border = './datasets/raw_data/label/border_change'
path_gray_mask = './datasets/raw_data/label/gray_mask'
seperate_path = './datasets/raw_data'
npy_path = './label/npy'

def input_rename(path, input_rename_path):
    input_list = os.listdir(path)
    input_list = [file for file in input_list if file.endswith('.png')]

    count = 0
    for image in input_list:
        input_image = cv2.imread(os.path.join(path, image),cv2.IMREAD_COLOR)
        input_name = 'input_%04d.png'% count
        count += 1

        cv2.imwrite(os.path.join(input_rename_path, input_name), input_image)

def border_pixel_change(image_list, path_mask, save_path):

    count = 0
    for img in image_list:
        image = cv2.imread(os.path.join(path_mask, img), cv2.IMREAD_COLOR)

        start = 0
        end = 2999

        h, w = image.shape[0], image.shape[1]

        # 테두리 pixel 값 바꾸기

        # right_low
        image[2999, 2999] = image[2998, 2998]
        # right_up
        image[0, 2999] = image[1, 2998]
        # left_low
        image[2999, 0] = image[2998, 1]
        # left_up
        image[0, 0] = image[1, 1]

        for i in range(h):
            if i == 0:
                continue
            image[i, 0] = image[i, 1]
            image[0, i] = image[1, i]
            image[i, 2999] = image[i, 2998]
            image[2999, i] = image[2998, i]

        # binary
        for i in range(h):
            for j in range(w):
                aa = image[i, j]
                threshold = aa[2]
                if aa[2] == 81:
                    image[i, j] = 255
                elif aa[0] == 230:
                    image[i, j] = 0

        cv2.imwrite(save_path+'/border_%03d.png'%count, image)
        count += 1

def gray_binary_change(image_list, border_path, gray_path):
    count = 0
    for i in image_list:
        image_rgn = cv2.imread(os.path.join(border_path, i))
        image_gray = cv2.cvtColor(image_rgn, cv2.COLOR_BGR2GRAY)

        gray_image_name = 'mask_%04d.png' % count
        count += 1

        cv2.imwrite(os.path.join(gray_path, gray_image_name), image_gray)

def npy_save(image_list, gray_path, npy_path):
    count = 0
    for i in image_list:
        mask_image = cv2.imread(os.path.join(gray_path, i))
        mask_gary = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_array = np.array(mask_gary)
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_array = mask_array / 255
        mask_name = 'label_%03d.npy'% count
        count += 1

        np.save(os.path.join(npy_path, mask_name), mask_array)

def train_valid_test_split(total_num, path):

    train_num = math.ceil((8 * total_num) / 10)
    v_t_total_num = total_num - int(train_num)
    validation_num = math.ceil((v_t_total_num / 2))
    test_num = math.floor((v_t_total_num / 2))

    input_path = path + '/input/images_rename/'
    label_path = path + '/label/gray_mask/'
    train_input_dir = path +'/train/input/'
    train_mask_dir = path +'/train/mask/'
    valid_input_dir = path +'/valid/input/'
    valid_mask_dir = path +'/valid/mask/'
    test_input_dir = path +'/test/input/'
    test_mask_dir = path +'/test/mask/'

    input_npy_list = os.listdir(input_path)
    mask_npy_list = os.listdir(label_path)

    image_file_list = [file for file in input_npy_list if file.endswith('.png')]
    label_file_list = [file for file in mask_npy_list if file.endswith('.png')]

    image_num = len(image_file_list)

    id_frame = np.arange(image_num)
    np.random.shuffle(id_frame)

    flag_num = 0

    for i in range(int(train_num)):
        current_image = image_file_list[id_frame[i + flag_num]]
        current_label = label_file_list[id_frame[i + flag_num]]

        current_image = cv2.imread(input_path + current_image, cv2.IMREAD_COLOR)
        current_label = cv2.imread(label_path + current_label, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(train_input_dir + 'input_train_%04d.png' % i, current_image)
        cv2.imwrite(train_mask_dir + 'mask_train_%04d.png' % i, current_label)

    flag_num += int(train_num)

    for i in range(int(validation_num)):
        current_image = image_file_list[id_frame[i + flag_num]]
        current_label = label_file_list[id_frame[i + flag_num]]

        current_image = cv2.imread(input_path + current_image, cv2.IMREAD_COLOR)
        current_label = cv2.imread(label_path + current_label, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(valid_input_dir + 'input_valid_%04d.png' % (i+flag_num), current_image)
        cv2.imwrite(valid_mask_dir + 'mask_valid_%04d.png' % (i+flag_num), current_label)

    flag_num += int(validation_num)

    for i in range(int(test_num)):
        current_image = image_file_list[id_frame[i + flag_num]]
        current_label = label_file_list[id_frame[i + flag_num]]

        current_image = cv2.imread(input_path + current_image, cv2.IMREAD_COLOR)
        current_label = cv2.imread(label_path + current_label, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(test_input_dir + 'input_test_%04d.png' % (i+flag_num), current_image)
        cv2.imwrite(test_mask_dir + 'mask_test_%04d.png' % (i+flag_num), current_label)

    print("################################## DATA Preprocessing FINISH ######################################")

# current_image = np.array(current_image)
# current_label = np.array(current_label)

if __name__ == '__main__':

    # mask image preprocessing (border pixel change)
    mask_list = os.listdir(path_mask)
    mask_list = [file for file in mask_list if file.endswith('.png')]
    border_pixel_change(mask_list, path_mask, save_path= path_border)

    # mask image preprocessing (gray_image transform)
    border_list = os.listdir(path_border)
    border_list = [file for file in border_list if file.endswith('.png')]
    gray_binary_change(border_list, path_border, path_gray_mask)

    # input image preprocessing (.png to .npy)
    input_rename(input_path, input_rename_path)

    # # seperate train, valid, test data set (8 : 1 : 1)
    # data_list = os.listdir(input_rename_path)
    # data_list = [file for file in data_list if file.endswith('.png')]
    # total_num = len(data_list)
    #
    # train_valid_test_split(total_num, seperate_path)




























# # mask image preprocessing (.png to .npy)
# gray_mask_list = os.listdir('./label/gray_mask')
# gray_mask_list = [file for file in gray_mask_list if file.endswith('.png')]
# npy_save(gray_mask_list, path_gray_mask, npy_path)
