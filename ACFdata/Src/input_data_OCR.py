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
        B = []
        label_B = []
        D = []
        label_D = []
        H = []
        label_H = []
        G = []
        label_G = []
        One = []
        label_One = []
        Two = []
        label_Two = []
        Three = []
        label_Three = []
        Four = []
        label_Four = []
        Five = []
        label_Five = []
        Six = []
        label_Six = []
        Seven = []
        label_Seven = []


        
        for file in os.listdir(file_dir):
            name = file.split(sep='_')
            if name[0] == 'B':
                B.append(file_dir+file)
                label_B.append(0)
            elif name[0] == 'D':
                D.append(file_dir+file)
                label_D.append(1)
            elif name[0] == 'H':
                H.append(file_dir+file)
                label_H.append(2)
            elif name[0] == 'G':
                G.append(file_dir+file)
                label_G.append(3)
            elif name[0] == '1':
                One.append(file_dir+file)
                label_One.append(4)
            elif name[0] == '2':
                Two.append(file_dir+file)
                label_Two.append(5)
            elif name[0] == '3':
                Three.append(file_dir+file)
                label_Three.append(6)
            elif name[0] == '4':
                Four.append(file_dir+file)
                label_Four.append(7)
            elif name[0] == '5':
                Five.append(file_dir+file)
                label_Five.append(8)
            elif name[0] == '6':
                Six.append(file_dir+file)
                label_Six.append(9)
            elif name[0] == '7':
                Seven.append(file_dir+file)
                label_Seven.append(10)

                
            else:
                pass
        
        print('There are %d B;\nThere are %d D;\nThere are %d G;\nThere are %d H;\nThere are %d One;\nThere are %d Two;\nThere are %d Three;\nThere are %d Four;\nThere are %d Five;\nThere are %d Six;\nThere are %d Seven;\n'
              %(len(B),len(D),len(G),len(H),len(One),len(Two),len(Three),len(Four),len(Five),len(Six),len(Seven)))
        
        image_list = np.hstack((B,D,G,H,One,Two,Three,Four,Five,Six,Seven))
        label_list = np.hstack((label_B,label_D,label_G,label_H,label_One,label_Two,label_Three,label_Four,label_Five,label_Six,label_Seven))
        
        temp = np.array([image_list,label_list])
        temp = temp.transpose()
        #np.random.shuffle(temp)
        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        label_list = [int(i) for i in label_list]
        
        label_list=tf.one_hot(label_list,11,1,0,-1)
#        
#        with tf.Session() as sess:
#            label=sess.run(label_list)
#            print(label)
    return image_list, label_list  
#    return label_list          
            

def get_batch(image,label,image_W,image_H,image_Channel,batch_size,shuffle=True,number_thread=1000,capacity=1000):
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

    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
        
        #make an input queue
    input_queue = tf.train.slice_input_producer([image,label],shuffle=shuffle)
        
    label =input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents,channels=image_Channel)
        
    #    image = tf.image.resize_image_with_crop_or_pad(image,image_H,image_W)
    image = float(255)-tf.image.resize_images(image,[image_H,image_W],method=3)
#        image_min = tf.reduce_min(image)
#        image_max = tf.reduce_max(image)
#        image = 125*(image - image_min)/(image_max-image_min)
#    image = tf.image.resize_images(image,[image_H,image_W],method=3)
        
    image = tf.image.per_image_standardization(image) #减去均值，除以方差，弄成正太分布
        
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size= batch_size,
                                              num_threads= number_thread, 
                                              capacity = capacity)
        
    label_batch = tf.reshape(label_batch, [batch_size,11])
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



    
