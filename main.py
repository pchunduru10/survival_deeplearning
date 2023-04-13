"""
Main script to train Survival CNN model.
NOTE: Make sure the packages are installed in the virtual environment before executing the model.
"""

import os,sys
sys.path.append('./utils/')
sys.path.append('./models/SCNN/')

import pandas as pd 
import numpy as np 
import datetime
import pickle

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from train_model import train


# Define Flags for hyperparamters 

tf.app.flags.DEFINE_float('learning_rate_top', 0.0001, 'Learning rate first few layers for adam optimizer')
## ONly one learning to test the model lr_top
tf.app.flags.DEFINE_float('learning_rate_bottom', 0.000001, 'Learning rate for final layers adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 18,34,50')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 256, 'Number of classes')
tf.app.flags.DEFINE_list('img_ch', [256,256], 'Input image dimension')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', '', 'Finetuning layers, seperated by commas')# fc,fc2,cox | "fc,scale5/block3,scale5/block2"#cox,fc1,scale5/block3,scale5/block2
tf.app.flags.DEFINE_string('optimizer', 'Adam', 'Optimizers to be used: SGD, Adam, Adagrad, Momentum')
tf.app.flags.DEFINE_boolean('pretrain', False, 'Continuing training from previous checkpoint')
tf.app.flags.DEFINE_boolean('augmenter', True, 'Do we need to augment training set')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate')
tf.app.flags.DEFINE_integer('required_improvement', 30, 'Early stopping to prevent overfitting')
tf.app.flags.DEFINE_string('data_path', '/data/filename.pickle', 'path to dataset')
tf.app.flags.DEFINE_string('tensorboard_root_dir', 'experiments', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


if __name__ == "__main__":
    tf.app.run()
    train(train_data= "dict object containing images, survival time and censoring as keys", 
        val_data= "dict object containing images, survival time and censoring as keys",
        test_data= None,
        FLAGS=FLAGS,
        weights_path= "path/to/pretrained/weights/ or None",
        pretrain_ckpt=None)
