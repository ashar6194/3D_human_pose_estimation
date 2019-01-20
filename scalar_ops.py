from classifier import color_mask
import numpy as np
import tensorflow as tf
import config

str_file_name = config.get_str_file_name(config.working_dataset)
with open(str_file_name, 'rb') as f2:
    med_freq_bal = np.float32(([l.strip() for l in f2]))
# print med_freq_bal.shape


def accuracy(logits, labels):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)

    shape = logits.get_shape().as_list()
    n = shape[3]

    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
    total_pixels = reduce(lambda x, y: x * y, shape[:3])
    return equal_pixels / total_pixels


def per_class_accuracy(logits, labels):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)
    shape = logits.get_shape().as_list()
    n = shape[3]

    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    equal_pixels = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(one_hot, labels), tf.equal(labels, 1))), [2, 1, 0])
    total_pixels = tf.reduce_sum(tf.to_float(tf.equal(labels, 1.0)), [2, 1, 0])
    return equal_pixels / total_pixels


def loss(logits, labels):
    class_weights = med_freq_bal * 100

    weights = tf.reduce_sum(class_weights * labels, axis=3)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    weighted_losses = cross_entropy * weights
    return tf.reduce_mean(weighted_losses, name='loss')

