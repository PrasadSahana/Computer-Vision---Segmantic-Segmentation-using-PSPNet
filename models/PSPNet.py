#The main model we are using for our projec
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from ModelBuilders import BackboneModels
import os, sys

def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)
def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):         
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)     #
    return net
def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d(net, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
    return net
def InterpBlock(net, level, feature_map_shape, pooling_type):
    kernel_size = [int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level)))]
    stride_size = kernel_size
    net = slim.pool(net, kernel_size, stride=stride_size, pooling_type='MAX')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = Upsampling(net, feature_map_shape)
    return net
def PyramidPoolingModule(inputs, feature_map_shape, pooling_type):
    interp_block1 = InterpBlock(inputs, 1, feature_map_shape, pooling_type)
    interp_block2 = InterpBlock(inputs, 2, feature_map_shape, pooling_type)
    interp_block3 = InterpBlock(inputs, 3, feature_map_shape, pooling_type)
    interp_block6 = InterpBlock(inputs, 6, feature_map_shape, pooling_type)
    res = tf.concat([inputs, interp_block6, interp_block3, interp_block2, interp_block1], axis=-1)
    return res
def build_pspnet(inputs, label_size, num_classes, preset_model='PSPNet', frontend="ResNet101", pooling_type = "MAX", #
    weight_decay=1e-5, upscaling_method="conv", is_training=True, pretrained_dir="models"):
    logits, end_points, frontend_scope, init_fn  = BackboneModels.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)
    feature_map_shape = [int(x / 8.0) for x in label_size]
    print(feature_map_shape)
    psp = PyramidPoolingModule(end_points['pool3'], feature_map_shape=feature_map_shape, pooling_type=pooling_type)
    net = slim.conv2d(psp, 512, [3, 3], activation_fn=None)
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    if upscaling_method.lower() == "conv":
        net = ConvUpscaleBlock(net, 256, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 256)
        net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 128)
        net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
        net = ConvBlock(net, 64)
    elif upscaling_method.lower() == "bilinear":
        net = Upsampling(net, label_size)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net, init_fn
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('length should match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)