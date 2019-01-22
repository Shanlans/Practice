# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
from PIL import Image
import os
import shutil
import time
import cv2
import scipy.misc


import numpy as np
import BasicLayerDef as layer
import input_data_activate_learning as input_data
from scipy.misc import toimage

import tensorflow.contrib.slim as slim


##Define Parameter##

DATE = '20171130'
DELETE = False

#Input Parameter#
IMAGE_WIDTH = 32;
IMAGE_HEIGHT = 32;  
IMAGE_CHANNEL = 1;

#Inference Parameter#
size_in = 0;
size_out= 0;
CLASS_NUM = 4;

#Training Parameter#
LR = 0.001
keep_prob=0.5
BATCH_SIZE = 64
VALIDATE_BATCH_SIZE = 160
TEST_BATCH_SIZE = 90
CAPACITY = 2000
MAX_STEP = 4000

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 5
CONV2_KENEL_NUM = 50
CONV2_KENEL_SIZE = 5
CONV3_KENEL_NUM = 70
CONV3_KENEL_SIZE = 5
CONV4_KENEL_NUM =  4
CONV4_KENEL_SIZE = 3



Kmin = 5 
accThreshold = 0.82
stopThreshold = 20


### 1.Creat data ### 
#number 1 to 10 data
train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_AL_data\\ACF_train\\'
train_pool_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_AL_data\\ACF_train_pool\\'
validate_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_validate\\'
test_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_test\\'
train_temp_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_AL_data\\ACF_train_temp\\'
logs_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_TrainLogs\\ACF_activate_learning\\20\\'
model_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_Models\\ACF_activate_learning\\20\\'


if DELETE == True:
    shutil.rmtree(logs_train_dir,ignore_errors=False, onerror=None)
    shutil.rmtree(model_train_dir,ignore_errors=False, onerror=None)
else:
    pass

train,train_label = input_data.get_files(train_dir)

train_batch, train_label_batch,_ = input_data.get_batch(train,
                                                      train_label,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_CHANNEL,
                                                      BATCH_SIZE, 
                                                      CAPACITY) 


train_pool,train_label_pool = input_data.get_files(train_pool_dir)

train_pool_batch, train_pool_label_batch,image_list = input_data.get_batch(train_pool,
                                                                           train_label_pool,
                                                                           IMAGE_WIDTH,
                                                                           IMAGE_HEIGHT,
                                                                           IMAGE_CHANNEL,
                                                                           len(train_pool), 
                                                                           CAPACITY) 


validate,validate_label = input_data.get_files(validate_dir)

validate_batch, validate_label_batch,_ = input_data.get_batch(validate,
                                                            validate_label,
                                                            IMAGE_WIDTH,
                                                            IMAGE_HEIGHT,
                                                            IMAGE_CHANNEL,
                                                            VALIDATE_BATCH_SIZE, 
                                                            CAPACITY) 

test,test_label = input_data.get_files(test_dir)

test_batch, test_label_batch,_ = input_data.get_batch(test,
                                                    test_label,
                                                    IMAGE_WIDTH,
                                                    IMAGE_HEIGHT,
                                                    IMAGE_CHANNEL,
                                                    TEST_BATCH_SIZE, 
                                                    CAPACITY) 





### 2.Define placeholder for inputs to network ###

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,32,32,1],name='Images') # 不规定有多少个图片’None‘，但是每一个图片都有784个点
    ys = tf.placeholder(tf.float32,[None,4],name='Labels') # 不规定有多少个输出’None‘，但是每个输出都是10个点（0-9） 
    trainphase = tf.placeholder(tf.bool,name='trainphase')
    tf.summary.histogram('inputs',xs)
    tf.summary.histogram('labels',ys)


### 3. Setup Network ###

# conv1 layer ##
BN0 = layer.batch_norm_layer(xs,trainphase,'BN0')

size_in = IMAGE_CHANNEL
size_out = CONV1_KENEL_NUM 
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv1,w1,_,_ = layer.conv_layer_withtah(inputs=BN0,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,name='conv1')
print('conv1 shape= ', conv1.get_shape())
## pool1 layer ##

drop1 = layer.dropoff(conv1,keep_prob=keep_prob,name='drop1')

BN1 = layer.batch_norm_layer(drop1,trainphase,'BN1')

size_in = CONV1_KENEL_NUM
size_out= CONV1_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool1,_,_ = layer.max_pool_2x2(inputs=BN1,name='maxpool1')
print('pool1 shape= ', pool1.get_shape())


## conv2 layer ##
size_in = CONV1_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv2,w2,_,_ = layer.conv_layer_withtah(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,name='conv2')
print('conv2 shape= ', conv2.get_shape())

#drop2 = layer.dropoff(conv2,keep_prob=keep_prob,name='drop2')
#
#BN2 = layer.batch_norm_layer(drop2,trainphase,'BN2')
## pool2 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool2,_,_ = layer.max_pool_2x2(inputs=conv2,name='maxpool2')
print('pool2 shape= ', pool2.get_shape())


## conv3 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv3,w3,_,_ = layer.conv_layer_withtah(inputs=pool2,size_in=size_in,size_out=size_out,kernel_size=CONV3_KENEL_SIZE,name='conv3')
print('conv3 shape= ', conv3.get_shape())

#drop3 = layer.dropoff(conv3,keep_prob=keep_prob,name='drop3')

#BN3 = layer.batch_norm_layer(drop3,trainphase,'BN3')

## pool3 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool3,_,_ = layer.max_pool_2x2(inputs=conv3,name='maxpool3')
print('pool3 shape= ', pool3.get_shape())

## conv4 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV4_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv4,w4,_,_ = layer.conv_layer_withtah(inputs=pool3,size_in=size_in,size_out=size_out,kernel_size=CONV4_KENEL_SIZE,name='conv4')
print('conv4 shape= ', conv4.get_shape())

#BN3 = layer.batch_norm_layer(conv3,trainphase,'BN3')

height = 4
with tf.name_scope('GAP'):
    glpooling = tf.nn.avg_pool(conv4,ksize=[1,height,height,1],strides=[1,height,height,1],padding='VALID')
    tf.summary.histogram('GAP',glpooling)
print('GAP shape= ', glpooling.get_shape())

## flat layer ##
size_in = CONV4_KENEL_NUM
size_out= CLASS_NUM
fc2 = tf.reshape(glpooling,shape=[-1,CLASS_NUM])
tf.summary.histogram('prediction',fc2)
print('fc2 shape= ', fc2.get_shape())



with tf.name_scope('Softmax'):
    prediction = tf.nn.softmax(fc2)
    
print('prediction shape= ', prediction.get_shape())

print('labels shape= ', train_label_batch.get_shape())
   


### 4. The error between prediction and real data ###
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=fc2,weights=1))# softmax + cross_entropy for classification      
    tf.summary.scalar('cross_entropy',cross_entropy)


### 5. Training Setting ###

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

### 6. Initial Variable ###

init = tf.global_variables_initializer()
#
#
#### 7. Start to training ###

def evaluation(logits, labels):
  with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
      tf.summary.scalar('accuracy', accuracy)
  return accuracy


train_acc = evaluation(prediction,ys)
validate_acc = evaluation(prediction,ys)
test_acc = evaluation(prediction,ys)


def record_image(inputs,split_num,outputs_num,name):
    dim = len(inputs.get_shape())-1
    image_list = tf.split(inputs,split_num,dim)
    name_number = 0 
    image = []
    for i in image_list:
        image_name = name + str(name_number)
        image.append(tf.summary.image(image_name,i,outputs_num))
        name_number+=1
    return image

def save_image(inputs,split_num):
    dim = len(inputs.get_shape())-1
    image_list = tf.split(inputs,split_num,dim)
    return image_list

def merge_image(inputs,batch_number,imgW,imgH,imageWidth,imageHight,row_number,path):    
    toImage = Image.new('L',(imageWidth,imageHight))
    image_group = [i[batch_number-1] for i in inputs ]
    i = 0 
    for image in image_group:
        image = image.reshape((imgW,imgH))
        loc = (((i % row_number) * imgW),(int(i/row_number) * imgW))
        image1 = toimage(image)
        toImage.paste(image1, loc)
        i+=1
    toImage.save(path)
    
def move_train_pool(file_list,index,prob_index):
    newfilelist=[]
    newindex=[]
    seen = set()
    for i,ele in enumerate(file_list[index]):
        if ele not in seen:
            newfilelist.append(ele)
            newindex.append(i)
            seen.add(ele)
    print(index)
    print(newindex)
    newindex=[index[i] for i in newindex]
    print(index)
    print('These file needs to be confirm: \n')
    for i in newfilelist:        
#        path=os.path.normpath(i)
        path = i.decode('UTF-8')
        print(path)
        shutil.move(path, train_temp_dir)
    print('These file are predicted to: \n')
    for j in prob_index[newindex]:
        if j == 0 :
            print('Depth')
        elif j == 1:
            print('Nan')
        elif j == 2:
            print('Ok')
        elif j == 3:
            print('Shallow')
    
    
    

### Training ##
#
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    
    w1 = tf.reshape(w1,shape=[CONV1_KENEL_NUM,CONV1_KENEL_SIZE,CONV1_KENEL_SIZE,IMAGE_CHANNEL])
    
    c1_image =[]
    c1_image = record_image(conv1,CONV1_KENEL_NUM,1,'c1_')
    c1_save_image = []
    c1_save_image = save_image(conv1,CONV1_KENEL_NUM)
    
    c2_image =[]
    c2_image = record_image(conv2,CONV2_KENEL_NUM,1,'c2_')
    c2_save_image = []
    c2_save_image = save_image(conv2,CONV2_KENEL_NUM)
    
    c3_image =[]
    c3_image = record_image(conv3,CONV3_KENEL_NUM,1,'c3_')
    c3_save_image = []
    c3_save_image = save_image(conv3,CONV3_KENEL_NUM)
               
    c4_image =[]
    c4_image = record_image(conv4,CONV4_KENEL_NUM,1,'c4_')
    c4_save_image = []
    c4_save_image = save_image(conv4,CONV4_KENEL_NUM)
    
    ip_image = tf.summary.image('Input',xs,max_outputs=4)
    w1_image = tf.summary.image('W1',w1,max_outputs=CONV1_KENEL_NUM)
    
    train_writer = tf.summary.FileWriter(logs_train_dir+'/train',sess.graph)  
    validate_writer = tf.summary.FileWriter(logs_train_dir+'/validate') 
    saver  = tf.train.Saver()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    train_stop_flag = 0
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_imgs,tra_lbls = sess.run([train_batch, train_label_batch])
            
#            _, tra_loss, tra_acc, Pre, Lable = sess.run([train_step,cross_entropy,train_acc,prediction,train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_step,cross_entropy,train_acc],feed_dict={xs:tra_imgs,ys:tra_lbls,trainphase:True})
            
#            sess.run(train_step,feed_dict={xs:tra_imgs,ys:tra_lbls})
            if step % 50 == 0:
                print('Step %d, train loss = %.4f, train accuracy = %.4f%%' %(step, tra_loss, tra_acc*100.0))
                val_imgs,val_lbls = sess.run([validate_batch, validate_label_batch])
                val_loss,val_acc = sess.run([cross_entropy,validate_acc],feed_dict={xs:val_imgs,ys:val_lbls,trainphase:False})
                print('Step %d, validation loss = %.4f, validation accuracy = %.4f%%' %(step, val_loss, val_acc*100.0))
                
                cv1_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:True}) for i in c1_image]               
                for i in cv1_image_group:
                    train_writer.add_summary(i,step)
                    
                cv2_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:True}) for i in c2_image]               
                for i in cv2_image_group:
                    train_writer.add_summary(i,step)
                
                cv3_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:True}) for i in c3_image]               
                for i in cv3_image_group:
                    train_writer.add_summary(i,step)
                    
                cv4_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:True}) for i in c4_image]               
                for i in cv4_image_group:
                    train_writer.add_summary(i,step)
                                        
                    
                summary_train = sess.run(merged,feed_dict={xs:tra_imgs,ys:tra_lbls,trainphase:True})
                train_writer.add_summary(summary_train, step)
                summary_val = sess.run(merged,feed_dict={xs:val_imgs,ys:val_lbls,trainphase:False})
                validate_writer.add_summary(summary_val, step)
                
                if val_acc < accThreshold:
                    train_stop_flag+=1
                               
#            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_train_dir, 'ACF_%s.ckpt'%DATE)
                saver.save(sess, checkpoint_path, global_step=step)
            if  (step + 1) == MAX_STEP or train_stop_flag>stopThreshold :
                cv1_save_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:False}) for i in c1_save_image]
                cv2_save_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:False}) for i in c2_save_image]
                cv3_save_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:False}) for i in c3_save_image]
                cv4_save_image_group = [sess.run(i,feed_dict={xs:tra_imgs,trainphase:False}) for i in c4_save_image]
                image_save_path = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\Src\\ACF_activate_learning\\image\\'
                cv1_image_name = image_save_path + 'cv1_tan.png'
                cv2_image_name = image_save_path + 'cv2_tan.png'
                cv3_image_name = image_save_path + 'cv3_tan.png'
                cv4_image_name = image_save_path + 'cv4_tan.png'
                orignal_image = image_save_path + 'original'+str(tra_lbls[0])+'.png'
                scipy.misc.imsave(orignal_image, tra_imgs[0].reshape((32,32)))
                merge_image(cv1_save_image_group,1,32,32,160,128,5,cv1_image_name)
                merge_image(cv2_save_image_group,1,16,16,160,80,10,cv2_image_name)
                merge_image(cv3_save_image_group,1,8,8,80,40,10,cv3_image_name)   
                merge_image(cv4_save_image_group,1,4,4,16,4,4,cv4_image_name)                          
                tst_imgs,tst_lbls = sess.run([test_batch, test_label_batch])
                start_time = time.time()
                tst_acc,Pre = sess.run([test_acc,prediction],feed_dict={xs:tst_imgs,ys:tst_lbls,trainphase:False}) 
                elapsed_time = (time.time() - start_time)/BATCH_SIZE               
                tra_pool_imgs,tra_pool_lbls,tra_pool_img_list = sess.run([train_pool_batch,train_pool_label_batch,image_list])
                tra_pool_prb= sess.run([prediction],feed_dict={xs:tra_pool_imgs,trainphase:False})
                tra_pool_prb= np.array(tra_pool_prb)
                tra_pool_prb= tra_pool_prb.reshape(tra_pool_prb.shape[1],tra_pool_prb.shape[2])
                tra_pool_prb_index = np.argmax(tra_pool_prb,axis=1) 
                tra_pool_prb_confid= tra_pool_prb[range(tra_pool_prb.shape[0]),tra_pool_prb_index]
                
                
                
                if len(train_pool)==0:
                    min_Index = 0 
                elif len(train_pool)<Kmin:
                    min_Index = np.argpartition(tra_pool_prb_confid, Kmin)[:len(train_pool)]
                elif len(train_pool)>=Kmin:
                    min_Index = np.argpartition(tra_pool_prb_confid, Kmin)[:Kmin]
                
                print(os.getcwd())
                move_train_pool(tra_pool_img_list,min_Index,tra_pool_prb_index)                                          
                print('The final test accuracy = %.4f%%' %(tst_acc*100.0))
                print('The average processing time is %.5f'%elapsed_time)
                if train_stop_flag > stopThreshold :
                    print('Please retrain')
                    break 
                               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
              
