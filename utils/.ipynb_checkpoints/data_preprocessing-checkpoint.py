import tensorflow as tf
import numpy as np


def resize_images(X_imgs, target_height,target_width ,CH):
    
    X_resized = []
    X_org = tf.placeholder(tf.float32, shape=(X_imgs[0].shape[0], X_imgs[0].shape[1], CH))  
    
    tf_resize =  tf.image.resize(X_org, size= (target_height, target_width), method= tf.image.ResizeMethod.BILINEAR ,
                                           preserve_aspect_ratio=False, name=None)
    #tf.image.resize_with_crop_or_pad(X_org,target_height, target_width)
    with tf.Session() as sess:
        for img in X_imgs:
            resized_img = sess.run([tf_resize], feed_dict={X_org: img})        
            X_resized.extend(resized_img)
    
    X_resized = np.array(X_resized, dtype=np.float32)
    return X_resized
     

def random_batch_augmentation(X_imgs, IMAGE_SIZE,CH):
    
    X_transformed = []
    X_org = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, CH))   

    # we want consistent augmentation for all images in a batch 
    x = tf.image.rot90(X_org, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    do_flip = tf.random_uniform([]) > 0.5
    x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: tf.image.flip_up_down(x))
    x = tf.cond(do_flip, 
                lambda: tf.image.random_contrast(x, lower = 0.2, upper = 1.8), 
                lambda: tf.image.random_saturation(x, lower=0.5, upper=1.5))
    x = tf.image.random_brightness(x, max_delta = 63)
#     x = tf.clip_by_value(x, 0, 1)#255)
    with tf.Session() as sess:
        tf.compat.v1.set_random_seed(123)
        for img in X_imgs:
            out = sess.run([x], feed_dict={X_org: img})
            X_transformed.extend(out)
    X_transformed = np.array(X_transformed, dtype=np.float32)
    return X_transformed

def normalize_images(imgs):
    ## Mean extraction images 
    mean_color = []
    std_color = []
    for i in range(0, 3):
        avg = np.mean(imgs[:, :, :, i])
        std = np.std(imgs[:, :, :, i])
        mean_color.append(avg)
        std_color.append(std)
    
    return mean_color,std_color

# def normalize_images(X_imgs,CH):
#     img_normalized = []
#     img_original = tf.placeholder(tf.float32,shape=(X_imgs[0].shape[0], X_imgs[0].shape[1], CH))

#     tf_normalized = tf.image.per_image_standardization(img_original)
#     with tf.Session() as sess:
#         for img in X_imgs:
#             res = sess.run([tf_normalized], feed_dict={img_original: img})        
#             img_normalized.extend(res)

#     img_normalized = np.array(img_normalized, dtype=np.float32) # Use of addtional memory 
#     return img_normalized
