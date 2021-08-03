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
import scipy.misc as misc

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("Epoch", "100", "Total Epoch")
tf.flags.DEFINE_integer("patch_size", "768", "patch size")
tf.flags.DEFINE_integer("overlay", "256", "overlay")
tf.flags.DEFINE_string("logs_dir", "log", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "datasets", "path to data directory")
tf.flags.DEFINE_string("gpu_idx", "0", "gpu_idx = 0")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test")
tf.flags.DEFINE_string('optimal_model', "model.ckpt-5", "ckpt optimal model")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 384
input_size = 3000

train_input_path = os.path.join(FLAGS.data_dir, 'train/input/training')
train_mask_path = os.path.join(FLAGS.data_dir, 'train/mask/training')
validation_input_path = os.path.join(FLAGS.data_dir, 'valid/input')
validation_mask_path = os.path.join(FLAGS.data_dir, 'valid/mask')
test_input_path = os.path.join(FLAGS.data_dir, 'test/input')
test_mask_path = os.path.join(FLAGS.data_dir, 'test/mask')

test_path = FLAGS.data_dir + '/test/'

valid_pred_image_path = './pred_mask/valid/pred'
valid_anno_image_path = './pred_mask/valid/annotation'
test_pred_image_path = './pred_mask/test/pred'
test_anno_image_path = './pred_mask/test/annotation'

Load_model_path = os.path.join(FLAGS.logs_dir, 'saver')
Load_model_path = os.path.join(Load_model_path, FLAGS.optimal_model)

patch_num = ut.patch_num(input_size, FLAGS.patch_size, FLAGS.overlay)

def shuffle(inputs, masks):
    # Shuffle the data
    perm = np.arange(inputs.shape[0])
    np.random.shuffle(perm)
    images = inputs[perm]
    labels = masks[perm]
    del inputs
    del masks
    gc.collect()

    return images, labels

def _transform(filename, __channels, image_options):
    # make sure images are of shape(h,w,3)
    image = misc.imread(filename)
    if __channels and len(image.shape) < 3:  
        image = np.array([image for i in range(3)])

    # resize image, mask -> nearest
    if image_options.get("resize", False) and image_options["resize"]:
        resize_size = int(image_options["resize_size"])
        resize_image = misc.imresize(image,
                                     [resize_size, resize_size], interp='nearest')
    else:
        resize_image = image

    return np.array(resize_image, dtype=np.int32)


def main(argv=None):
    with tf.device('/gpu:' + str(FLAGS.gpu_idx)):
        # place holder parameters
        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
        annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

        # argmax -> pred_annotation, layer output -> logits, loss
        pred_annotation, logits = m.inference(image, keep_probability, FLAGS.debug, FLAGS.model_dir, NUM_OF_CLASSESS)

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

            # resize,
            image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
            __channels = True
            inputs_train = np.array(
                [_transform(os.path.join(train_input_path, file), __channels, image_options) for file in
                 input_list_train])
            __channels = False
            masks_train = np.array(
                [_transform(os.path.join(train_mask_path, file), __channels, image_options) for file in
                 mask_list_train], dtype=np.int32)
            masks_train = np.expand_dims(masks_train, axis=-1)
            masks_train = masks_train / 255
            masks_train = np.array(masks_train, dtype=np.int32)

            # resize,
            __channels = True
            inputs_valid = np.array(
                [_transform(os.path.join(validation_input_path, file), __channels, image_options) for file in
                 input_list_valid])
            __channels = False
            masks_valid = np.array(
                [_transform(os.path.join(validation_mask_path, file), __channels, image_options) for file in
                 mask_list_valid], dtype=np.int32)
            masks_valid = np.expand_dims(masks_valid, axis=-1)
            masks_valid = masks_valid / 255
            masks_valid = np.array(masks_valid, dtype=np.int32)

            print(f"inputs train shape : {inputs_train.shape}")
            print(f"masks train shape : {masks_train.shape}")
            print(f"inputs valid shape: {inputs_valid.shape}")
            print(f"masks valid shape : {masks_valid.shape}")

            # final train, valid cost and epoch
            minimum_train_cost = float('inf')
            validation_cost_flag = float('inf')
            epoch_train = 0
            epoch_valid = 0

            count_patch_train = 0
            count_patch_valid = 0

            for epoch in range(FLAGS.Epoch):

                inputs_train, masks_train = shuffle(inputs_train, masks_train)

                file_size = inputs_train.shape[0]
                num_batches = file_size // FLAGS.batch_size

                # one epoch cost save
                cost_batch = 0
                cost_image = 0
                summary_train_cost_result = 0

                for i in range(num_batches):
                    start_idx = i * FLAGS.batch_size
                    end_idx = (i + 1) * FLAGS.batch_size

                    feed_dict = {image: inputs_train[start_idx:end_idx,:,:,:], annotation: masks_train[start_idx:end_idx,:,:,:], keep_probability: 0.85}

                    train_val, cost, summary_val = sess.run([train_op, loss, loss_summary], feed_dict=feed_dict)

                    cost_batch += cost

                    train_writer.add_summary(summary_val, count_patch_train)
                    count_patch_train += 1


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

                file_size_valid = inputs_valid.shape[0]
                num_batches_valid = file_size_valid // FLAGS.batch_size

                for i in range(num_batches_valid):
                    start_idx_valid = i * FLAGS.batch_size
                    end_idx_valid = (i + 1) * FLAGS.batch_size

                    valid_cost_summary, valid_cost, valid_pred_mask, valid_accuracy = sess.run([loss_summary, loss, pred_annotation, accuracy],
                                                                                               feed_dict={image: inputs_valid[start_idx_valid:end_idx_valid,:,:,:], annotation: masks_valid[start_idx_valid:end_idx_valid,:,:,:],
                                                                                                          keep_probability: 1.0})

                    cost_valid_batch += valid_cost

                    # tensorboard valid cost save
                    validation_writer.add_summary(valid_cost_summary, count_patch_valid)
                    count_patch_valid += 1

                    accuracy_valid_avg += valid_accuracy
                    count_ac += 1

                    # visualization
                    if epoch == 0:
                        ut.anno_visualization(valid_anno_image_path, inputs_valid[start_idx_valid:end_idx_valid,:,:,:], masks_valid[start_idx_valid:end_idx_valid,:,:,:], count)
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

            # resize,
            image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
            __channels = True
            inputs_test = np.array([_transform(os.path.join(test_input_path, file), __channels, image_options) for file in input_list_test])
            __channels = False
            masks_test = np.array([_transform(os.path.join(test_mask_path, file), __channels, image_options) for file in mask_list_test])
            masks_test = np.expand_dims(masks_test, axis=-1)
            masks_test = masks_test / 255
            masks_test = np.array(masks_test, dtype=np.int32)

            print(f"inputs train shape : {inputs_test.shape}")
            print(f"masks train shape : {masks_test.shape}")

            print("!!test start!!")
            # accuracy avg count
            accuracy_test_avg = 0
            count = 0

            # visualization count
            count_v = 0

            file_num_test = inputs_test.shape[0]
            batch_num_test = file_num_test // FLAGS.batch_size

            for i in range(batch_num_test):
                start_idx_test = i * FLAGS.batch_size
                end_idx_test = (i+1)* FLAGS.batch_size


                test_pred_mask, test_accuracy = sess.run([pred_annotation, accuracy],
                                                         feed_dict={image: inputs_test[start_idx_test:end_idx_test,:,:,:], annotation: masks_test[start_idx_test:end_idx_test,:,:,:], keep_probability: 1.0})

                accuracy_test_avg += test_accuracy
                count += 1
                print('Accuracy : {}'.format(test_accuracy))

                # 3000 * 3000 resize


                # visualization
                ut.anno_visualization(test_anno_image_path, inputs_test[start_idx_test:end_idx_test,:,:,:], masks_test[start_idx_test:end_idx_test,:,:,:], count_v)
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