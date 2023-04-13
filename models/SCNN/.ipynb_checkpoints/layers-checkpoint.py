import tensorflow as tf 
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

## This 
CONV_WEIGHT_DECAY = 0.00004 #0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01

def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)


def convolution(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
  
def fully_connected(x, num_units_out):
    num_units_in = x.get_shape()[1]
#     x = tf.reshape(bottom, [-1, in_size])
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_DECAY)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    return tf.nn.xw_plus_b(x, weights, biases)

def batch_normalization(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x


def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]
    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = convolution(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = batch_normalization(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = convolution(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = batch_normalization(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = convolution(b, ksize=1, stride=1, filters_out=filters_out)
        c = batch_normalization(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = convolution(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = batch_normalization(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)
    