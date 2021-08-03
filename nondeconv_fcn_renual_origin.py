from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
from six.moves import xrange
import gc
import os
import cv2
import scipy.misc as misc
import random
from sklearn.utils import shuffle

import data_load as DATA_LOAD
import util as ut
import model as m
import TensorflowUtils as utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_integer("Epoch", "30", "Total Epoch")
tf.flags.DEFINE_integer("patch_size", "768", "patch size")
tf.flags.DEFINE_integer("overlay", "256", "overlay")
tf.flags.DEFINE_string("logs_dir", "log", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "datasets", "path to data directory")
tf.flags.DEFINE_string("gpu_idx", "0", "gpu_idx = 0")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test")
tf.flags.DEFINE_string('optimal_model', "model.ckpt-21", "ckpt optimal model")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 3
IMAGE_SIZE = 1024
input_size = 3000

train_input_path = './datasets/raw_data/train/aug_input'
train_mask_path = './datasets/raw_data/train/aug_mask'
validation_input_path = './datasets/raw_data/valid/input'
validation_mask_path = './datasets/raw_data/valid/mask'
test_input_path = './datasets/raw_data/test/input'
test_mask_path = './datasets/raw_data/test/mask'

test_path = FLAGS.data_dir + '/test/'

valid_pred_image_path = './pred_mask/valid/pred'
valid_anno_image_path = './pred_mask/valid/annotation'
test_pred_image_path = './pred_mask/test/pred'
test_anno_image_path = './pred_mask/test/annotation'

Load_model_path = os.path.join(FLAGS.logs_dir, 'saver')
Load_model_path = os.path.join(Load_model_path, FLAGS.optimal_model)

patch_num = ut.patch_num(input_size, FLAGS.patch_size, FLAGS.overlay)

def resize_image_3000(label, target_height=3000, target_width=3000):
    label = tf.image.resize(label, [target_height, target_width], method='nearest')
    return label


def resize_img(image_options, __channels, input_path, mask_path, input_list_train, mask_list_train):

    # resize array create
    inputs_train = np.array(
        [_transform_input(os.path.join(input_path, file), __channels, image_options) for file in
         input_list_train])
    __channels = False
    masks_train = np.array(
        [_transform_label(os.path.join(mask_path, file), __channels, image_options) for file in
         mask_list_train], dtype=np.int32)
    masks_train = np.expand_dims(masks_train, axis=-1)
    masks_train = masks_train / 100
    masks_train = np.array(masks_train, dtype=np.int32)

    return inputs_train, masks_train

def shuffle_list(inputs_list, masks_list):
    # Shuffle the data
    inputs_list, masks_list = shuffle(inputs_list, masks_list)

    return inputs_list, masks_list

def _transform_label(filename, __channels, image_options):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if __channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        image = np.array([image for i in range(3)])

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,
                                     [resize_size, resize_size], interp='nearest')
    else:
        resize_image = image

    return np.array(resize_image, dtype=np.int32)

def _transform_input(filename, __channels, image_options):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if __channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        image = np.array([image for i in range(3)])

    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,
                                     [resize_size, resize_size], interp='nearest')
    else:
        resize_image = image

    return np.array(resize_image, dtype=np.int32)


def main(argv=None):
    with tf.device('/gpu:' + str(FLAGS.gpu_idx)):
        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        is_train = tf.placeholder(tf.bool, name="is_train")
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

        # argmax -> pred_annotation, layer output -> logits, loss
        pred_annotation, logits = m.inference(image, keep_probability, FLAGS.debug, FLAGS.model_dir, NUM_OF_CLASSESS, is_train)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    # accuracy
    correct_prediction = tf.equal(x=tf.argmax(logits, -1, output_type=tf.int32),
                                  y=tf.squeeze(annotation, squeeze_dims=[3]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 저장한 weight, bias 불러오기
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            # variable histogram tensorboard
            utils.add_to_regularization_and_summary(var)
    # cost minimize optimizer
    train_op = m.train(loss, trainable_var, FLAGS.learning_rate, FLAGS.debug)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/tensorboard/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/tensorboard/validation')

    sess.run(tf.global_variables_initializer())

    # ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir+'/saver/')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print("Model restored...")

    if FLAGS.mode == "train":
        try:
            # data load
            # data_load (train), read image, read mask
            input_list_train = os.listdir(train_input_path)
            input_list_train = [file for file in input_list_train if file.endswith("png")]
            mask_list_train = os.listdir(train_mask_path)
            mask_list_train = [file for file in mask_list_train if file.endswith('png')]

            input_list_valid = os.listdir(validation_input_path)
            input_list_valid = [file for file in input_list_valid if file.endswith("png")]
            mask_list_valid = os.listdir(validation_mask_path)
            mask_list_valid = [file for file in mask_list_valid if file.endswith('png')]

            print(f"inputs train data num : {len(input_list_train)}")
            print(f"masks train data num : {len(mask_list_train)}")
            print(f"inputs valid data num : {len(input_list_valid)}")
            print(f"masks valid data num : {len(mask_list_valid)}")

            # final train, valid cost and epoch
            minimum_train_cost = float('inf')
            validation_cost_flag = float('inf')
            epoch_train = 0
            epoch_valid = 0

            count_patch_train = 0
            count_patch_valid = 0

            for epoch in range(FLAGS.Epoch):

                inputs_train, masks_train = shuffle_list(input_list_train, mask_list_train)

                file_size = len(inputs_train)
                num_batches = file_size // FLAGS.batch_size

                # one epoch cost save
                cost_batch = 0
                cost_image = 0
                summary_train_cost_result = 0

                for i in range(num_batches):
                    start_idx = i * FLAGS.batch_size
                    end_idx = (i + 1) * FLAGS.batch_size

                    # path batch size index load
                    inputs_list_train = inputs_train[start_idx:end_idx]
                    masks_list_train = masks_train[start_idx:end_idx]

                    # resize array create
                    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
                    __channels = True
                    inputs_train_feed, masks_train_feed = resize_img(image_options, __channels, train_input_path, train_mask_path, inputs_list_train, masks_list_train)

                    # augmentation code 수정 필요.
                    # inputs_train_feed, masks_train_feed = ut.augmentation_r_f_f(inputs_train_feed, masks_train_feed)

                    # feed dict parameters
                    feed_dict = {image: inputs_train_feed, annotation: masks_train_feed, keep_probability: 0.85, is_train:True}

                    train_val, cost, summary_val = sess.run([train_op, loss, loss_summary], feed_dict=feed_dict)

                    cost_batch += cost

                    train_writer.add_summary(summary_val, count_patch_train)
                    count_patch_train += 1

                    del inputs_train_feed, masks_train_feed
                    gc.collect()

                # batch cost calcul
                cost_total_train = cost_batch / float(num_batches)

                print(f"epoch : {epoch}, train_cost : {cost_total_train}")

                # minimum train cost, epoch
                if minimum_train_cost > cost_total_train:
                    minimum_train_cost = cost_total_train
                    epoch_train = epoch


                print("validation!!!")

                # visualization total image num
                count = 0
                count_ac = 0
                accuracy_valid_avg = 0

                # one epoch cost save
                cost_valid_batch = 0
                cost_valid_image = 0
                batch_valid_num = 0
                summary_valid_cost_result = 0

                file_size_valid = len(input_list_valid)
                num_batches_valid = file_size_valid // FLAGS.batch_size

                for i in range(num_batches_valid):
                    start_idx_valid = i * FLAGS.batch_size
                    end_idx_valid = (i + 1) * FLAGS.batch_size

                    inputs_list_valid = input_list_valid[start_idx_valid:end_idx_valid]
                    masks_list_valid = mask_list_valid[start_idx_valid:end_idx_valid]

                    # resize array create
                    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
                    __channels = True
                    inputs_valid_feed, masks_valid_feed = resize_img(image_options, __channels, validation_input_path, validation_mask_path, inputs_list_valid, masks_list_valid)

                    valid_cost_summary, valid_cost, valid_pred_mask, valid_accuracy = sess.run([loss_summary, loss, pred_annotation, accuracy],
                                                                                               feed_dict={image: inputs_valid_feed, annotation: masks_valid_feed,
                                                                                                          keep_probability: 1.0, is_train:False})

                    cost_valid_batch += valid_cost

                    # tensorboard valid cost save
                    validation_writer.add_summary(valid_cost_summary, count_patch_valid)
                    count_patch_valid += 1

                    accuracy_valid_avg += valid_accuracy
                    count_ac += 1

                    # visualization
                    if epoch == 0:
                        ut.anno_visualization(valid_anno_image_path, inputs_valid_feed, masks_valid_feed, count)
                        ut.pred_valid_visualization(epoch, valid_pred_image_path, valid_pred_mask, count)
                        count += FLAGS.batch_size
                    else:
                        ut.pred_valid_visualization(epoch, valid_pred_image_path, valid_pred_mask, count)
                        count += FLAGS.batch_size

                # accuracy
                accuracy_valid_epoch = accuracy_valid_avg / float(count_ac)
                print(f'epoch : {epoch}/{FLAGS.Epoch - 1}, validation Accuracy : {accuracy_valid_epoch}')

                # valid total cost
                cost_total_valid = cost_valid_batch / float(num_batches_valid)

                print(f"Epoch : {epoch}/{FLAGS.Epoch - 1}, total_cost_valid : {cost_total_valid}")

                # validation cost update
                if cost_total_valid < validation_cost_flag:
                    validation_cost_flag = cost_total_valid
                    epoch_valid = epoch
                    print(f"valid final update cost : {validation_cost_flag}, epoch {epoch}, result ckpt.model : model_{epoch}")

                    saver.save(sess, './log/saver/model.ckpt', global_step=epoch)
                else:
                    print("Not Update Valid Loss")

            print(f"final minimum train cost / epoch : {minimum_train_cost} / {epoch_train}")
            print(f"final minimum valid cost / epoch : {validation_cost_flag} / {epoch_valid}")

        except Exception as e:
            print(f"Exception : {e}")

    elif FLAGS.mode == "test":

        try:
            # saved optimal model load
            saver.restore(sess, Load_model_path)

            # data load
            # data_load (train), read image, read mask
            input_list_test = os.listdir(test_input_path)
            input_list_test = [file for file in input_list_test if file.endswith("png")]
            mask_list_test = os.listdir(test_mask_path)
            mask_list_test = [file for file in mask_list_test if file.endswith('png')]

            print(f"inputs train shape : {len(input_list_test)}")
            print(f"masks train shape : {len(mask_list_test)}")

            print("!!test start!!")
            # accuracy avg count
            accuracy_test_avg = 0
            count = 0

            # visualization count
            count_v = 0

            file_num_test = len(input_list_test)
            batch_num_test = file_num_test // FLAGS.batch_size

            for i in range(batch_num_test):
                start_idx_test = i * FLAGS.batch_size
                end_idx_test = (i+1)* FLAGS.batch_size

                inputs_list_test = input_list_test[start_idx_test:end_idx_test]
                masks_list_test = mask_list_test[start_idx_test:end_idx_test]

                # resize array create
                image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
                __channels = True
                inputs_test_feed, masks_test_feed = resize_img(image_options, __channels, test_input_path,
                                                                 test_mask_path, inputs_list_test,
                                                                 masks_list_test)

                test_pred_mask, test_accuracy = sess.run([pred_annotation, accuracy],
                                                         feed_dict={image: inputs_test_feed, annotation: masks_test_feed, keep_probability: 1.0, is_train:False})

                accuracy_test_avg += test_accuracy
                count += 1
                print('Accuracy : {}'.format(test_accuracy))

                # visualization
                ut.anno_visualization(test_anno_image_path, inputs_test_feed, masks_test_feed, count_v)
                ut.pred_test_visualization(test_pred_image_path, test_pred_mask, count_v)
                count_v += FLAGS.batch_size

            accuracy_test_avg = accuracy_test_avg / float(count)
            print(f"Final TEST AVG Accuracy : {round(accuracy_test_avg, 4)}")
        except Exception as e:
            print(f"Exception : {e}")

if __name__ == "__main__":
    tf.app.run()


















    #     W8 = utils.weight_variable([1, 1, 1024, 512], name="W8")
    #     b8 = utils.bias_variable([512], name="b8")
    #     conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
    #     # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
    #
    #     # size : 48 filter 512
    #     upsampling_1 = upsampling2d(conv8, size=(2, 2), name='up1')
    #     fuse_1 = tf.add(upsampling_1, pool5, name='fuse_1')
    #
    #     # size : 96 filter 512
    #     upsampling_2 = upsampling2d(fuse_1, size=(2, 2), name='up2')
    #     fuse_2 = tf.add(upsampling_2, image_net['pool4'], name='fuse_2')
    #
    #     # size : 192 filter 256
    #     upsampling_3 = upsampling2d(fuse_2, size=(2, 2), name='up3')
    #     fuse_3 = tf.add(upsampling_3, image_net['pool3'], name='fuse_3')
    #
    #     # filter 384 filter 128
    #     upsampling_4 = upsampling2d(fuse_3, size=(2, 2), name='up4')
    #     fuse_4 = tf.add(upsampling_4, image_net['pool2'], name='fuse_4')
    #
    #     # filter 768 filter 64
    #     upsampling_5 = upsampling2d(fuse_4, size=(2, 2), name='up5')
    #     fuse_5 = tf.add(upsampling_5, image_net['pool1'], name='fuse_5')
    #
    #     # final conv
    #     W9 = utils.weight_variable([1, 1, 64, NUM_OF_CLASSESS], name="W8")
    #     b9 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
    #     conv9 = utils.conv2d_basic(fuse_5, W9, b9)
    #
    #     annotation_pred = tf.argmax(conv9, dimension=3, name="prediction")
    #
    # return tf.expand_dims(annotation_pred, dim=3), conv9