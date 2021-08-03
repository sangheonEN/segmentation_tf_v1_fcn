import tensorflow as tf
import TensorflowUtils as utils
import numpy as np

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

def upsampling2d(x, size=(2, 2), name='upsampling2d'):
    with tf.name_scope(name):
        shape = x.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))

def concat(values, axis, name='concat'):
    output = tf.concat(values=values, axis=axis, name=name)

    return output

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def vgg_net(weights, image, debug):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")#매틀랩 conv layer shape -> tensorflow 의 conv layer shape으로 맞추기 위해 transpose
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob, debug, model_dir, NUM_Class, is_train):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    # vgg net model 불러오기
    model_data = utils.get_model_data(model_dir, MODEL_URL)

    # model_data['normalization'] pixel 값 중 가장 큰값??
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    # image pixel - mean pixel (pixel normalization)
    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image, debug)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        # 224 일때 conv5_3 size -> h, w : (7, 7) 786 일때 conv5_3 size -> h, w : (24, 24)
        W6 = utils.weight_variable([12, 12, 512, 1024], name="W6")
        b6 = utils.bias_variable([1024], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        batch_norm_6 = batch_norm(conv6, 1024, is_train)
        relu6 = tf.nn.relu(batch_norm_6, name="relu6")
        if debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 1024, 1024], name="W7")
        b7 = utils.bias_variable([1024], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        batch_norm_7 = batch_norm(conv7, 1024, is_train)
        relu7 = tf.nn.relu(batch_norm_7, name="relu7")
        if debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 1024, NUM_Class], name="W8")
        b8 = utils.bias_variable([NUM_Class], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        batch_norm_8 = batch_norm(conv8, NUM_Class, is_train)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")


        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_Class], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(batch_norm_8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_Class])
        W_t3 = utils.weight_variable([16, 16, NUM_Class, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_Class], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list, learning_rate, debug):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)



# interporation upsampling

# import tensorflow as tf
# import TensorflowUtils as utils
# import numpy as np
#
# MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
#
# def upsampling2d(x, size=(2, 2), name='upsampling2d'):
#     with tf.name_scope(name):
#         shape = x.get_shape().as_list()
#         return tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))
#
# def concat(values, axis, name='concat'):
#     output = tf.concat(values=values, axis=axis, name=name)
#
#     return output
#
# def batch_norm(x, n_out, phase_train):
#     """
#     Batch normalization on convolutional maps.
#     Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#     Args:
#         x:           Tensor, 4D BHWD input maps
#         n_out:       integer, depth of input maps
#         phase_train: boolean tf.Varialbe, true indicates training phase
#         scope:       string, variable scope
#     Return:
#         normed:      batch-normalized maps
#     """
#     with tf.variable_scope('bn'):
#         beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                                      name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                                       name='gamma', trainable=True)
#         batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(phase_train,
#                             mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
#
# def vgg_net(weights, image, debug):
#     layers = (
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
#         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
#         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
#         'relu5_3', 'conv5_4', 'relu5_4'
#     )
#
#     net = {}
#     current = image
#     for i, name in enumerate(layers):
#         kind = name[:4]
#         if kind == 'conv':
#             kernels, bias = weights[i][0][0][0][0]
#             # matconvnet: weights are [width, height, in_channels, out_channels]
#             # tensorflow: weights are [height, width, in_channels, out_channels]
#             kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")#매틀랩 conv layer shape -> tensorflow 의 conv layer shape으로 맞추기 위해 transpose
#             bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
#             current = utils.conv2d_basic(current, kernels, bias)
#         elif kind == 'relu':
#             current = tf.nn.relu(current, name=name)
#             if debug:
#                 utils.add_activation_summary(current)
#         elif kind == 'pool':
#             current = utils.avg_pool_2x2(current)
#         net[name] = current
#
#     return net
#
#
# def inference(image, keep_prob, debug, model_dir, NUM_Class, is_train):
#     """
#     Semantic segmentation network definition
#     :param image: input image. Should have values in range 0-255
#     :param keep_prob:
#     :return:
#     """
#     print("setting up vgg initialized conv layers ...")
#     # vgg net model 불러오기
#     model_data = utils.get_model_data(model_dir, MODEL_URL)
#
#     # model_data['normalization'] pixel 값 중 가장 큰값??
#     mean = model_data['normalization'][0][0][0]
#     mean_pixel = np.mean(mean, axis=(0, 1))
#
#     weights = np.squeeze(model_data['layers'])
#
#     # image pixel - mean pixel (pixel normalization)
#     processed_image = utils.process_image(image, mean_pixel)
#
#     with tf.variable_scope("inference"):
#         image_net = vgg_net(weights, processed_image, debug)
#         conv_final_layer = image_net["conv5_3"]
#
#         pool5 = utils.max_pool_2x2(conv_final_layer)
#
#         # 224 일때 conv5_3 size -> h, w : (7, 7) 786 일때 conv5_3 size -> h, w : (24, 24)
#         W6 = utils.weight_variable([12, 12, 512, 1024], name="W6")
#         b6 = utils.bias_variable([1024], name="b6")
#         conv6 = utils.conv2d_basic(pool5, W6, b6)
#         batch_norm_6 = batch_norm(conv6, 1024, is_train)
#         relu6 = tf.nn.relu(batch_norm_6, name="relu6")
#         if debug:
#             utils.add_activation_summary(relu6)
#         relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
#
#         W7 = utils.weight_variable([1, 1, 1024, 1024], name="W7")
#         b7 = utils.bias_variable([1024], name="b7")
#         conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
#         batch_norm_7 = batch_norm(conv7, 1024, is_train)
#         relu7 = tf.nn.relu(batch_norm_7, name="relu7")
#         if debug:
#             utils.add_activation_summary(relu7)
#         relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
#
#         W8 = utils.weight_variable([1, 1, 1024, NUM_Class], name="W8")
#         b8 = utils.bias_variable([NUM_Class], name="b8")
#         conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
#         batch_norm_8 = batch_norm(conv8, NUM_Class, is_train)
#         # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
#
#         # size : 48 filter 512
#         upsampling_1 = upsampling2d(batch_norm_8, size=(2, 2), name='up1')
#         concat_1 = concat([upsampling_1, image_net['pool4']], axis=3, name='concat_1')
#
#         # size : 96 filter 512
#         upsampling_2 = upsampling2d(concat_1, size=(2, 2), name='up2')
#         concat_2 = concat([upsampling_2, image_net['pool3']], axis=3, name='concat_2')
#
#         # size : 192 filter 256
#         upsampling_3 = upsampling2d(concat_2, size=(2, 2), name='up3')
#         concat_3 = concat([upsampling_3, image_net['pool2']], axis=3, name='concat_3')
#
#         # filter 384 filter 128
#         upsampling_4 = upsampling2d(concat_3, size=(2, 2), name='up4')
#         concat_4 = concat([upsampling_4, image_net['pool1']], axis=3, name='concat_4')
#
#         # filter 768 filter 64
#         upsampling_5 = upsampling2d(concat_4, size=(2, 2), name='up5')
#         concat_5 = concat([upsampling_5, image_net['conv1_2']], axis=3, name='concat_5')
#
#         # final conv
#         shape = concat_5.get_shape()
#         W9 = utils.weight_variable([1, 1, shape[3].value, NUM_Class], name="W9")
#         b9 = utils.bias_variable([NUM_Class], name="b9")
#         conv9 = utils.conv2d_basic(concat_5, W9, b9)
#
#         annotation_pred = tf.argmax(conv9, dimension=3, name="prediction")
#
#     return tf.expand_dims(annotation_pred, dim=3), conv9
#
#
# def train(loss_val, var_list, learning_rate, debug):
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     grads = optimizer.compute_gradients(loss_val, var_list=var_list)
#     if debug:
#         # print(len(var_list))
#         for grad, var in grads:
#             utils.add_gradient_summary(grad, var)
#     return optimizer.apply_gradients(grads)




















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






















# import tensorflow as tf
# import TensorflowUtils as utils
# import numpy as np
#
# MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
#
# def upsampling2d(x, size=(2, 2), name='upsampling2d'):
#     with tf.name_scope(name):
#         shape = x.get_shape().as_list()
#         return tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))
#
# def concat(values, axis, name='concat'):
#     output = tf.concat(values=values, axis=axis, name=name)
#
#     return output
#
# def batch_norm(x, n_out, phase_train):
#     """
#     Batch normalization on convolutional maps.
#     Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#     Args:
#         x:           Tensor, 4D BHWD input maps
#         n_out:       integer, depth of input maps
#         phase_train: boolean tf.Varialbe, true indicates training phase
#         scope:       string, variable scope
#     Return:
#         normed:      batch-normalized maps
#     """
#     with tf.variable_scope('bn'):
#         beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                                      name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                                       name='gamma', trainable=True)
#         batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(phase_train,
#                             mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
#
# def vgg_net(weights, image, debug):
#     layers = (
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
#         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
#         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
#         'relu5_3', 'conv5_4', 'relu5_4'
#     )
#
#     net = {}
#     current = image
#     for i, name in enumerate(layers):
#         kind = name[:4]
#         if kind == 'conv':
#             kernels, bias = weights[i][0][0][0][0]
#             # matconvnet: weights are [width, height, in_channels, out_channels]
#             # tensorflow: weights are [height, width, in_channels, out_channels]
#             kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")#매틀랩 conv layer shape -> tensorflow 의 conv layer shape으로 맞추기 위해 transpose
#             bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
#             current = utils.conv2d_basic(current, kernels, bias)
#         elif kind == 'relu':
#             current = tf.nn.relu(current, name=name)
#             if debug:
#                 utils.add_activation_summary(current)
#         elif kind == 'pool':
#             current = utils.avg_pool_2x2(current)
#         net[name] = current
#
#     return net
#
#
# def inference(image, keep_prob, debug, model_dir, NUM_Class, is_train):
#     """
#     Semantic segmentation network definition
#     :param image: input image. Should have values in range 0-255
#     :param keep_prob:
#     :return:
#     """
#     print("setting up vgg initialized conv layers ...")
#     # vgg net model 불러오기
#     model_data = utils.get_model_data(model_dir, MODEL_URL)
#
#     # model_data['normalization'] pixel 값 중 가장 큰값??
#     mean = model_data['normalization'][0][0][0]
#     mean_pixel = np.mean(mean, axis=(0, 1))
#
#     weights = np.squeeze(model_data['layers'])
#
#     # image pixel - mean pixel (pixel normalization)
#     processed_image = utils.process_image(image, mean_pixel)
#
#     with tf.variable_scope("inference"):
#         image_net = vgg_net(weights, processed_image, debug)
#         conv_final_layer = image_net["conv5_3"]
#
#         pool5 = utils.max_pool_2x2(conv_final_layer)
#
#         # 224 일때 conv5_3 size -> h, w : (7, 7) 786 일때 conv5_3 size -> h, w : (24, 24)
#         W6 = utils.weight_variable([12, 12, 512, 1024], name="W6")
#         b6 = utils.bias_variable([1024], name="b6")
#         conv6 = utils.conv2d_basic(pool5, W6, b6)
#         relu6 = tf.nn.relu(conv6, name="relu6")
#         if debug:
#             utils.add_activation_summary(relu6)
#         relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
#
#         W7 = utils.weight_variable([1, 1, 1024, 1024], name="W7")
#         b7 = utils.bias_variable([1024], name="b7")
#         conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
#         relu7 = tf.nn.relu(conv7, name="relu7")
#         if debug:
#             utils.add_activation_summary(relu7)
#         relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
#
#         W8 = utils.weight_variable([1, 1, 1024, NUM_Class], name="W8")
#         b8 = utils.bias_variable([NUM_Class], name="b8")
#         conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
#         # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
#
#         # size : 48 filter 512
#         upsampling_1 = upsampling2d(conv8, size=(2, 2), name='up1')
#         concat_1 = concat([upsampling_1, image_net['pool4']], axis=3, name='concat_1')
#
#         # size : 96 filter 512
#         upsampling_2 = upsampling2d(concat_1, size=(2, 2), name='up2')
#         concat_2 = concat([upsampling_2, image_net['pool3']], axis=3, name='concat_2')
#
#         # size : 192 filter 256
#         upsampling_3 = upsampling2d(concat_2, size=(2, 2), name='up3')
#         concat_3 = concat([upsampling_3, image_net['pool2']], axis=3, name='concat_3')
#
#         # filter 384 filter 128
#         upsampling_4 = upsampling2d(concat_3, size=(2, 2), name='up4')
#         concat_4 = concat([upsampling_4, image_net['pool1']], axis=3, name='concat_4')
#
#         # filter 768 filter 64
#         upsampling_5 = upsampling2d(concat_4, size=(2, 2), name='up5')
#         concat_5 = concat([upsampling_5, image_net['conv1_2']], axis=3, name='concat_5')
#
#         # final conv
#         shape = concat_5.get_shape()
#         W9 = utils.weight_variable([1, 1, shape[3].value, NUM_Class], name="W9")
#         b9 = utils.bias_variable([NUM_Class], name="b9")
#         conv9 = utils.conv2d_basic(concat_5, W9, b9)
#
#         annotation_pred = tf.argmax(conv9, dimension=3, name="prediction")
#
#     return tf.expand_dims(annotation_pred, dim=3), conv9
#
#
# def train(loss_val, var_list, learning_rate, debug):
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     grads = optimizer.compute_gradients(loss_val, var_list=var_list)
#     if debug:
#         # print(len(var_list))
#         for grad, var in grads:
#             utils.add_gradient_summary(grad, var)
#     return optimizer.apply_gradients(grads)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     #     W8 = utils.weight_variable([1, 1, 1024, 512], name="W8")
#     #     b8 = utils.bias_variable([512], name="b8")
#     #     conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
#     #     # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
#     #
#     #     # size : 48 filter 512
#     #     upsampling_1 = upsampling2d(conv8, size=(2, 2), name='up1')
#     #     fuse_1 = tf.add(upsampling_1, pool5, name='fuse_1')
#     #
#     #     # size : 96 filter 512
#     #     upsampling_2 = upsampling2d(fuse_1, size=(2, 2), name='up2')
#     #     fuse_2 = tf.add(upsampling_2, image_net['pool4'], name='fuse_2')
#     #
#     #     # size : 192 filter 256
#     #     upsampling_3 = upsampling2d(fuse_2, size=(2, 2), name='up3')
#     #     fuse_3 = tf.add(upsampling_3, image_net['pool3'], name='fuse_3')
#     #
#     #     # filter 384 filter 128
#     #     upsampling_4 = upsampling2d(fuse_3, size=(2, 2), name='up4')
#     #     fuse_4 = tf.add(upsampling_4, image_net['pool2'], name='fuse_4')
#     #
#     #     # filter 768 filter 64
#     #     upsampling_5 = upsampling2d(fuse_4, size=(2, 2), name='up5')
#     #     fuse_5 = tf.add(upsampling_5, image_net['pool1'], name='fuse_5')
#     #
#     #     # final conv
#     #     W9 = utils.weight_variable([1, 1, 64, NUM_OF_CLASSESS], name="W8")
#     #     b9 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
#     #     conv9 = utils.conv2d_basic(fuse_5, W9, b9)
#     #
#     #     annotation_pred = tf.argmax(conv9, dimension=3, name="prediction")
#     #
#     # return tf.expand_dims(annotation_pred, dim=3), conv9