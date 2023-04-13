"""
Script to build and train  Survival CNN model using ResNet architecture.
NOTE: Make sure the packages are installed in the virtual environment before executing the model.
"""

import os,sys

import pandas as pd 
import pickle
import numpy as np 
import datetime
import math
from lifelines.utils import concordance_index
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from network import ResNetModel
from dataLoader import BatchPreprocessor
from utils import early_stop, losses, optimizers, dataLoader ,weights ,util,tracker

c = tf.ConfigProto()
c.gpu_options.allow_growth=True
c.gpu_options.per_process_gpu_memory_fraction = 0.95
c.allow_soft_placement = True
c.log_device_placement = False


def train(train_data= None, 
          val_data= None,
          test_data= None,
          FLAGS=None,
          weights_path=None ,
          pretrain_ckpt = None):
    
    ## paths & log dirs
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')
    
    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)
    
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate_top={}\n'.format(FLAGS.learning_rate_top))
    flags_file.write('learning_rate_bottom={}\n'.format(FLAGS.learning_rate_bottom))
    flags_file.write('input_image_size={}\n'.format(FLAGS.img_ch[0]))
    flags_file.write('dropout_rate={}\n'.format(FLAGS.dropout))
    flags_file.write('optimizer={}\n'.format(FLAGS.optimizer))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('required_improvement={}\n'.format(FLAGS.required_improvement))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('data_augmentation={}\n'.format(FLAGS.augmenter))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    ## Placeholders
    img = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.img_ch[0],  FLAGS.img_ch[1], 3],name="images")
    cens = tf.compat.v1.placeholder(dtype = tf.int32,name= 'censoring')
    riskset = tf.compat.v1.placeholder(dtype= tf.bool, shape =(None),name='at_risk_samples')
    
    is_training = tf.compat.v1.placeholder('bool', [])
    is_dropout = tf.compat.v1.placeholder('bool', [])
   

     ## training parameters 
    loader_train = dataLoader.__dict__['BatchPreprocessor'](data=train_data, shuffle=True,output_size=[FLAGS.img_ch[0],FLAGS.img_ch[1]],
                              batch_size =FLAGS.batch_size, normalize=True, augment = True ) 
    
    loader_val = dataLoader.__dict__['BatchPreprocessor'](data=val_data, shuffle=False,output_size=[FLAGS.img_ch[0],FLAGS.img_ch[1]],
                              batch_size =FLAGS.batch_size, normalize=True, augment = False )
    
   
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training,
                        is_dropout, 
                        num_classes=FLAGS.num_classes,
                        keep_prob= FLAGS.dropout)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    

    risk_output = model.network_fn(img)
    loss, pred = losses.__dict__['negative_loglike_loss'](cox_output =risk_output, censoring =cens,riskset=riskset)
    train_op_top,train_op_bottom = optimizers.__dict__['optimizer'](loss,FLAGS.learning_rate_top,FLAGS.learning_rate_bottom, 
                                                train_layers,global_step,name = FLAGS.optimizer)
    if train_layers is None: 
        train_layers = tf.trainable_variables().name
    trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
    var_list = [v for v in tf.trainable_variables() if
        v.name.split(':')[0].split('/')[-1] in trainable_var_names and
        util.contains(v.name, train_layers)]

    trainer = tf.group(train_op_bottom.minimize(loss, global_step=global_step,var_list= var_list[:len(var_list)-20]),
                       train_op_top.minimize(loss, global_step=global_step,var_list= var_list[len(var_list)-20:]))
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # tensrflow paramters
    train_summary = tf.summary.scalar('train_loss', loss)#tf.summary.merge_all()
#     val_summary = tf.summary.scalar('val_loss', loss)
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver(max_to_keep=25)
    patience = early_stop.early_stop(patience=FLAGS.required_improvement)
    
    trainingStats = {}
    trainingStats["training"]= {}
    trainingStats["validation"]={}
    trainingStats["training"]['data']= {}
    trainingStats["training"]["ci"] = {}
    trainingStats["validation"]["data"] = {}
    trainingStats["validation"]["ci"] = {}
   
    with tf.Session(config=c) as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(tensorboard_train_dir,sess.graph)
        if(FLAGS.pretrain):
            saver.restore(sess, tf.train.latest_checkpoint(pretrain_ckpt)) # search for checkpoint file
            graph = tf.get_default_graph()
            train_writer.add_graph(sess.graph)
        
        # Load the pretrained weights
        weights.load_pretrained_weights(weights_path,sess, skip_layers=[] )    #skip_layers=train_layers
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))
        
        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            print("{} Start training...".format(datetime.datetime.now()))
            train_track = tracker.tracker()
            val_track = tracker.tracker()
            while(loader_train.pointer < loader_train.data_size ):
                [batch_imgs, batch_cens, batch_time] = loader_train.fetch_batch()
                risk_set = util.make_riskset(batch_time)
                summary,train_loss,train_risk,_ = sess.run([train_summary,loss, pred, trainer], feed_dict={ img: batch_imgs,
                                                                                        cens: batch_cens,
                                                                                        riskset: risk_set,
                                                                                        is_training: True, 
                                                                                        is_dropout: True})
                train_track.loss_increment(train_loss)
                train_track.risk_increment(train_risk)
                if epoch !=0 and epoch % FLAGS.log_step == 0:
                    train_writer.add_summary(summary, sess.run(global_step)) # for on random batch in epoch
                    train_writer.flush()
                    
            print("Completed batch training for epoch {}".format(epoch))
            avg_train_loss= train_track.average(loader_train.batch_min)
            
            if(math.isnan(avg_train_loss)):
                print("Training loss is Nan")
                sess.close()
                return 
            
            train_data_reduced = util.median_cindex_data(risk= np.array(train_track.stored_risk).ravel(),
                                                        time= loader_train.time,
                                                        cens= loader_train.cens,
                                                        patient_ID = loader_train.patientID)
            
            ci_train = concordance_index(train_data_reduced['time'],
                                         -np.exp(train_data_reduced['risk']), 
                                         train_data_reduced['cens'])   
            
            print("{} Median Training Accuracy = {:.4f}".format(datetime.datetime.now(), ci_train))
            print("{} Total Training Losss = {:.4f}".format(datetime.datetime.now(), avg_train_loss))
            
            
            print("{} Start validation".format(datetime.datetime.now()))
            while(loader_val.pointer < loader_val.data_size ):
                [ batch_val_imgs, batch_val_cens, batch_val_time] = loader_val.fetch_batch()
                val_risk_set = util.make_riskset(batch_val_time)
                if(loader_val.data_size - loader_val.pointer >5 ):
                    val_loss,val_risk = sess.run([loss,pred ], feed_dict={ img: batch_val_imgs,
                                                                      cens: batch_val_cens,
                                                                      riskset: val_risk_set,
                                                                      is_training: False,
                                                                      is_dropout: False})
                    val_track.loss_increment(val_loss)
                    val_track.risk_increment(val_risk)
                else:
                    val_risk = sess.run(pred, feed_dict={ img: batch_val_imgs,
                                                                      cens: batch_val_cens,
                                                                      riskset: val_risk_set,
                                                                      is_training: False,
                                                                      is_dropout: False})
                     
                    val_track.risk_increment(val_risk)
            
            print("Completed validation for epoch {}".format(epoch))
            avg_val_loss= val_track.average(loader_val.batch_min)          
            val_data_reduced = util.median_cindex_data(risk= np.array(val_track.stored_risk).ravel(),
                                                        time= loader_val.time,
                                                        cens= loader_val.cens,
                                                        patient_ID = loader_val.patientID)
            
            ci_val = concordance_index(val_data_reduced['time'],
                                         -np.exp(val_data_reduced['risk']), 
                                         val_data_reduced['cens'])   
            print("{} Median Validation Accuracy = {:.4f}".format(datetime.datetime.now(), ci_val))
            print("{} Total Validation Losss = {:.4f}".format(datetime.datetime.now(), avg_val_loss))
                       
            ## Reset pointer for train and validation 
            loader_train.reset_pointer()
            loader_val.reset_pointer()
            
            save_flag, stop_flag = patience.track(avg_val_loss)
            
            if(FLAGS.num_epochs - epoch <=5 and stop_flag != False):
                trainingStats["training"]['data'][epoch] = train_data_reduced
                trainingStats["training"]['ci'][epoch]= ci_train
                trainingStats["validation"]['data'][epoch] = val_data_reduced
                trainingStats["validation"]['ci'][epoch] = ci_val
            
            if save_flag:
                print( '!!!New checkpoint at epoch: {}\t, Validation loss: {:.4f}\t'.format( iter, avg_val_loss ) )
                checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
                saver.save(sess, checkpoint_path)
                print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))  

            if stop_flag:
                print('Stopping model due to no improvement for {} validation runs'.format(FLAGS.required_improvement) )
                trainingStats["training"]['data'][epoch] = train_data_reduced
                trainingStats["training"]['ci'][epoch]= ci_train
                trainingStats["validation"]['data'][epoch] = val_data_reduced
                trainingStats["validation"]['ci'][epoch] = ci_val
                print("Saving model weights after max improvement threshold")
                weights.save_weights(sess,saved_weights)
                train_writer.close()
#                 val_writer.close()
                sess.close()
                return trainingStats
    
        weights.save_weights(sess,train_dir)
    ## save the final graph as txt file 
    tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)
    train_writer.close()

    save_output(trainingStats,train_dir, "training_stats")



def save_output(data: Any, filepath: str, filename: str = "train_history_dict"):
    """Save output as pickle file in given location.
    :param data: data to save (predictions or history)
    :type data: Any
    """
    
    with open(os.path.join(filepath, f"{filename}.p"), 'wb') as file_pi:
        pickle.dump(data, file_pi)






            
         
    

