# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os

#img_width = 28
#img_height = 28
#
#file_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_data'

def get_files(file_dir):
    with tf.name_scope('Get_data_labels'):
        Dianzi = []
        label_dianzi = []
        Zangwu = []
        label_zangwu = []

        
        for file in os.listdir(file_dir):
            name = file.split(sep='_')
            if name[0] == 'Dianzi':
                Dianzi.append(file_dir+file)
                label_dianzi.append(0)
            elif name[0] == 'Zangwu':
                Zangwu.append(file_dir+file)
                label_zangwu.append(1)
            else:
                pass
        
        print('There are %d Dianzi;\nThere are %d Zangwu;\n'
              %(len(Dianzi),len(Zangwu)))
        
        image_list = np.hstack((Dianzi,Zangwu))
        label_list = np.hstack((label_dianzi,label_zangwu))
        
        temp = np.array([image_list,label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        label_list = [int(i) for i in label_list]
        
        label_list=tf.one_hot(label_list,2,1,0,-1)
#        
#        with tf.Session() as sess:
#            label=sess.run(label_list)
#            print(label)
    return image_list, label_list  
#    return label_list          
            

def get_batch(image,label,image_W,image_H,image_Channel,batch_size,capacity):
    '''
    Args:
        image: list type
        label: list type
        image_H: image Height
        image_W: image Width
        image_Channel:image Channel
        batch_size: batch size
        capacity: the maximum elements in queue 
    Return:
        image batch: 4D tensor [batch_size, image_W, image_H, image_Channel],dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.float32
    '''
    with tf.name_scope('Generate_Batch'):
        image = tf.cast(image,tf.string)
        label = tf.cast(label,tf.int32)
        
        #make an input queue
        input_queue = tf.train.slice_input_producer([image,label])
        
        label =input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_png(image_contents,channels=image_Channel)
        
    #    image = tf.image.resize_image_with_crop_or_pad(image,image_H,image_W)
        image = tf.image.resize_images(image,[image_H,image_W])
        
        image = tf.image.per_image_standardization(image) #减去均值，除以方差，弄成正太分布
        
        image_batch, label_batch = tf.train.batch([image, label],
                                                    batch_size= batch_size,
                                                    num_threads= 10, 
                                                    capacity = capacity)
        
        label_batch = tf.reshape(label_batch, [batch_size,2])
        image_batch = tf.cast(image_batch, tf.float32)
        
#        with tf.Session() as sess:
#            label=sess.run(label_batch)
#            image=sess.run(image_batch)
#            print(label)
#            print(image)
    
    return image_batch, label_batch
    
    


#BATCH_SIZE = 10
#CAPACITY = 256
#IMG_W = 28
#IMG_H = 28
#IMG_C = 1
#
#train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_data\\'
#
#image_list, label_list = get_files(train_dir)
#image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H,IMG_C,BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,0])
#                plt.show()
#            i+=1
#            input()
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)



    
