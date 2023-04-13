import tensorflow as tf 
from utils import util 

LEARNING_DECAY= 1e-01
MOMENTUM =  0.9
MOVING_AVERAGE_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
L1_reg = 0.0
L2_reg = 0.01


def optimizer(loss,lr_top, lr_bottom, train_layers=[], global_step = None, name = None ):

    '''
    params train_layers :  layers of netowrk  to fine tune 
    params global_step : keeps track of number of batches passed 
    params lr_top : if train_layers is not specified, learning rate for top layers
    params lr_bottom : if train_layers is not specified, learning rate for bottom layers

    '''
#     if train_layers is None: 
#             train_layers = tf.trainable_variables().name
#     trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
#     var_list = [v for v in tf.trainable_variables() if
#         v.name.split(':')[0].split('/')[-1] in trainable_var_names and
#         util.contains(v.name, train_layers)]
        
    if(name == "Adam"):
        train_op_top = tf.train.AdamOptimizer(lr_top)
        train_op_bottom = tf.train.AdamOptimizer(lr_bottom)
#         train_op_top = tf.train.AdamOptimizer(lr_top).minimize(loss, global_step=global_step,var_list=var_list[:len(var_list)-10])
#         train_op_bottom = tf.train.AdamOptimizer(lr_bottom).minimize(loss, global_step=global_step,var_list=var_list[len(var_list)-10:])

    elif(name == "SGD"):
            # Update learning rate
        lr_top = tf.train.inverse_time_decay(learning_rate= lr_top,
                                                        global_step = global_step,
                                                        decay_steps = 1,
                                                        decay_rate = LEARNING_DECAY)
        lr_bottom = tf.train.inverse_time_decay(learning_rate= lr_bottom,
                                                        global_step = global_step,
                                                        decay_steps = 1,
                                                        decay_rate = LEARNING_DECAY) 

        train_op_top = tf.train.GradientDescentOptimizer(lr_top)
        train_op_bottom = tf.train.GradientDescentOptimizer(lr_bottom)

    elif(name == "Adagrad"):
        lr_top = tf.train.inverse_time_decay(learning_rate= lr_top,
                                                        global_step = global_step,
                                                        decay_steps = 1,
                                                        decay_rate = LEARNING_DECAY)
#         lr_bottom = tf.train.inverse_time_decay(learning_rate= lr_bottom,
#                                                         global_step = global_step,
#                                                         decay_steps = 1,
#                                                         decay_rate = LEARNING_DECAY) 
        train_op_top = tf.train.AdagradOptimizer(lr_top)
#         train_op_top = tf.train.AdagradOptimizer(lr_top).minimize(loss, global_step=global_step,var_list=var_list[:len(var_list)-10])
#         train_op_bottom = tf.train.AdagradOptimizer(lr_bottom).minimize(loss, global_step=global_step,var_list=var_list[len(var_list)-10:])
        
    else:
        train_op_top = tf.train.MomentumOptimizer(lr_top,momentum = MOMENTUM, use_nesterov =True)
        train_op_bottom = tf.train.MomentumOptimizer(lr_bottom,momentum = MOMENTUM, use_nesterov =True)
    
    return train_op_top












