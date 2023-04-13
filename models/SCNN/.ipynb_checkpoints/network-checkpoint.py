#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pchunduru
Implemented finetune network for ResNet architecutre with 50,101 and 152 blocks.


"""
import tensorflow as tf
from layers import convolution, batch_normalization,fully_connected, stack, block 

NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}


class ResNetModel(object):

    def __init__(self, 
                is_training =True,
                is_dropout =False ,
                num_classes = 1000,
                keep_prob=0.5): 
        """
        Implements ResNet50 architecture referenced from https://arxiv.org/pdf/1512.03385.pdf
        param is_training : 
        param is_Dropout : If dropout is required at subsequent layer
        param depth : Number of blocks for the architecture
        param keep_prob : Dropout values to be applied 
        """
    
        self.is_training = is_training
        self.is_dropout = is_dropout
        self.keep_prob =  keep_prob
        self.num_classes = num_classes
        depth  = 50 # change for other architectures 
        if depth in NUM_BLOCKS:
            self.num_blocks = NUM_BLOCKS[depth]
        else:
            raise ValueError('Depth is not supported; it must be 50, 101 or 152')

    def network_fn(self, image_tf):

        
        # Scale 1
        with tf.variable_scope('scale1'):
            s1_conv = convolution(image_tf, ksize= 7, stride=2, filters_out=64)
            s1_bn = batch_normalization(s1_conv, is_training=self.is_training)
            s1 = tf.nn.relu(s1_bn)
        
        # Scale 2
        with tf.variable_scope('scale2'):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, is_training=self.is_training, num_blocks=self.num_blocks[0], stack_stride= 1 , block_filters_internal=64)
                
        # Scale 3
        with tf.variable_scope('scale3'):
            s3 = stack(s2, is_training=self.is_training, num_blocks=self.num_blocks[1], stack_stride=2, block_filters_internal=128)
        
        # Scale 4
        with tf.variable_scope('scale4'):
            s4 = stack(s3, is_training=self.is_training, num_blocks=self.num_blocks[2], stack_stride=2, block_filters_internal=256)
        
        # Scale 5
        with tf.variable_scope('scale5'):
            s5 = stack(s4, is_training=self.is_training, num_blocks=self.num_blocks[3], stack_stride=2, block_filters_internal=512)
        

        # post-net
        avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')

        with tf.variable_scope('fc1'):        
            fc1 = fully_connected(avg_pool, num_units_out=1000) 
        
        fc1 = tf.cond(tf.equal(self.is_dropout, tf.constant(True)),
                                 lambda:tf.nn.dropout(fc1, self.keep_prob) , lambda:fc1)
        
        with tf.variable_scope('fc2'):        
            fc2 = fully_connected(fc1, num_units_out=self.num_classes) 
        
        fc2 = tf.cond(tf.equal(self.is_dropout, tf.constant(True)),
                                 lambda:tf.nn.dropout(fc2, self.keep_prob) , lambda:fc2)
         
                        
        # construct Cox layer
        with tf.variable_scope('risk_output'):
            risk_output = fully_connected(fc2, num_units_out=1)
            risk_output = tf.reshape(risk_output,[-1]) # -1 means "all"
           
        return risk_output

    