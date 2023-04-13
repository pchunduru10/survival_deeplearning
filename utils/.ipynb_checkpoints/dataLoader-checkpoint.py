import numpy as np
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
from data_preprocessing import random_batch_augmentation, resize_images,normalize_images

## pickle load
# def pklLoad(file):
#     with open(file, "rb") as f:
#         unpickler = pickle.Unpickler(f)
#         data_dict = unpickler.load()
#     return data_dict


class BatchPreprocessor():
    def __init__(self,
		data=None, 
		shuffle=False,
		output_size=[256,256],
		batch_size =None,
		normalize=True,
		augment = True ):
        
        """
		params 
		params
		params
		params
		params
		params

		"""
        self.output_size = output_size
        self.normalize = normalize
        self.augment =augment
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.images = []
        self.time = []
        self.cens =[]
        self.patientID = []
        self.pointer = 0

        self.images= data['images']
        self.cens = data['censoring']
        self.time = data['survival_time']
        self.patientID = data['ID']
        self.order_idx = np.arange(len(self.time)) 
        self.data_size = len(self.order_idx)
        self.batch_min = np.floor(self.data_size/self.batch_size)
#         self.batch_max = np.ceil(self.data_size/self.batch_size)
        
        if(self.normalize):
            self.mean_color,self.std_color = normalize_images(imgs=self.images)
        
        if(self.shuffle):
            self.shuffle_data()

    def shuffle_data(self):
        self.random_idx = np.random.permutation(len(self.time))
        
    def reset_pointer(self):
        self.pointer = 0
        if(self.shuffle):
            self.shuffle_data()
            
#     def fetch_batch(self):
#         if(self.batch_count ==0 and self.shuffle):
#                 self.shuffle_data()
#         for idx in range(self.batch_size):
#             if(idx + self.batch_size*self.batch_count) < self.data_size:
#                 if(self.shuffle):
#                     batch_idx = self.random_idx[idx + self.batch_size*self.batch_count]
#                 else:
#                     batch_idx = self.order_idx[idx + self.batch_size*self.batch_count]
#             else:
#                 if(self.shuffle):
#                 batch_idx = self.random_idx[self.pointer:]
#             else:
#                 batch_idx = self.order_idx[self.pointer:]
        
#         self.batch_count += 1
                
    
    def fetch_batch(self):    
# 		self.n_batches = np.floor(self.images.shape[0] / self.batch_size).astype(np.int16)
        if(self.pointer<int(self.batch_size*self.batch_min)):
            if(self.shuffle):
                batch_idx =self.random_idx[self.pointer:(self.pointer+self.batch_size)]
            else:
                batch_idx = self.order_idx[self.pointer:(self.pointer+self.batch_size)]
            self.pointer += self.batch_size # update pointer 
        else:
            if(self.shuffle):
                batch_idx = self.random_idx[self.pointer:]
            else:
                batch_idx = self.order_idx[self.pointer:]
            self.pointer += len(batch_idx)

        images = self.images[batch_idx]
        time = self.time.iloc[batch_idx].reset_index(drop=True)#total_batches*batch_size)
        cens = self.cens.iloc[batch_idx].reset_index(drop=True)
        
        if(self.normalize):            
            mean_scaled_images = np.ndarray([len(batch_idx), self.images.shape[1], self.images.shape[1], 3])
            for idx in range(images.shape[0]):
                img = images[idx]
                # Subtract mean color and divide by stanadard deviation 
                img_mean = img- np.array(self.mean_color)
                img_mean_std = img_mean/np.array(self.std_color)         
                mean_scaled_images[idx] = img_mean_std 
            images = mean_scaled_images
       
        if(self.augment):
        	## Resize or crop images
	        images = resize_images(images, self.output_size[0], self.output_size[1],3)
	        images = random_batch_augmentation(images, self.output_size[0],3)
        else:
            images = resize_images(images, self.output_size[0], self.output_size[1],3)

        return images,cens, time










		
	













