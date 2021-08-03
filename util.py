import os
import cv2
import numpy as np
import tensorflow as tf
import random

def patch_num(input_size, patch_size, overlay_size):
    # patch num count
    x = 0
    count_patch = 0
    for i in range(100):
        c = x + patch_size
        x += patch_size - overlay_size
        count_patch += 1
        if input_size < c:
            break

    return count_patch

def pred_valid_visualization(epoch_1, valid_pred_path, pred, count):

    valid_pred_dir_save_epoch = os.path.join(valid_pred_path, 'epoch_%d' % epoch_1)
    if not os.path.exists(valid_pred_dir_save_epoch):
        os.makedirs(valid_pred_dir_save_epoch)

    for i in range(pred.shape[0]):
        pred_mask = pred[i] * 100
        cv2.imwrite(valid_pred_dir_save_epoch + '/pred_mask_%03d_Epoch(%03d).png' % (count, epoch_1), pred_mask)
        count += 1

def pred_test_visualization(test_pred_path, pred, count):

    test_pred_dir_save_epoch = os.path.join(test_pred_path, 'test_pred')
    if not os.path.exists(test_pred_dir_save_epoch):
        os.makedirs(test_pred_dir_save_epoch)

    for i in range(pred.shape[0]):
        pred_mask = pred[i] * 100
        cv2.imwrite(test_pred_dir_save_epoch + '/pred_mask_%03d.png' %count, pred_mask)
        count += 1

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
        anno_mask = mask[i] * 100
        cv2.imwrite(valid_anno_mask_dir_save_epoch + '/annotation_mask_%03d.png' %count_m, anno_mask)
        count_m += 1

def data_shuffle(train_input_list, train_mask_list):
    train_images_num = len(train_input_list)

    id_frame = np.arange(train_images_num)
    np.random.shuffle(id_frame)

    train_inputs = []
    train_masks = []

    for i in range(train_images_num):
        train_inputs.append(train_input_list[id_frame[i]])
        train_masks.append(train_mask_list[id_frame[i]])

    return train_inputs, train_masks

def resize_image(
    image, label, target_height=1000, target_width=1000):
    image = tf.image.resize(image, [target_height, target_width])
    label = tf.image.resize(label, [target_height, target_width], method='nearest')
    return image, label

def random_flip_horizontal(image, label):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if random.uniform(0, 1) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    return image, label

def random_flip_vertical(image, label):
    """Flips image and boxes vertically with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if random.uniform(0, 1) > 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

def random_rot(image, label):
    """Rotates image and boxes with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if random.uniform(0, 1) > 0.5:
        image = tf.image.rot90(image)
        label = tf.image.rot90(label)
    return image, label

def resize_image(
    image, label, target_height=1000, target_width=1000):
    image = tf.image.resize(image, [target_height, target_width])
    label = tf.image.resize(label, [target_height, target_width], method='nearest')
    return image, label

def augmentation_r_f_f(images, labels):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = images
    label = labels
    image, label = resize_image(image, label, target_height=1024, target_width=1024)
    image, label = random_flip_horizontal(image, label)
    image, label = random_flip_vertical(image, label)
    image, label = random_rot(image, label)

    return image, label

