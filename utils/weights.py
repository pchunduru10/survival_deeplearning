import numpy as np
import tensorflow as tf
import datetime
'''
Loading and saving fine tuned model weights

'''

def save_weights(session,saved_weights_path):
        variables_names = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
        values= session.run(variables_names)
        data_dict = {}
        for name, val in zip(variables_names, values):
            if name not in data_dict:
                parts=name.split(":0")[0]
                data_dict[parts] = val
        
        ## save model weights 
        now = datetime.datetime.now()
        file_name = now.strftime('resnet50_finetuned_%Y%m%d_%H%M%S')
        path = saved_weights_path + file_name + ".npy"
        np.save(path, data_dict)
        print("Finetuned model weights saved at: {}".format(saved_weights_path))
    

def load_pretrained_weights(weights_path, session, skip_layers=[]):
        if weights_path is not None:
            print("Loading pretrained weights")
            weights_dict = np.load(weights_path, encoding='bytes',allow_pickle=True).item()

            for op_name in weights_dict:
                parts = op_name.split('/')
                if parts[0] == 'fc1' or parts[0] == 'fc'  or parts[0] == 'cox': # 'fc' for pre-trained weigths 
                    continue

                full_name = "{}:0".format(op_name)
                for v in tf.global_variables():
                        if v.name == full_name:
                            var = [v][0]
                session.run(var.assign(weights_dict[op_name]))
        else:
            session.run(tf.global_variables_initializer())
