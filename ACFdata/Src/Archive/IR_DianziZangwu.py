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
import input_data_ir_DianziZangwu as input_data
from scipy.misc import toimage

import tensorflow.contrib.slim as slim


##Define Parameter##

DATE = '20171220'
DELETE = False
BABYTRAIN = True
#Input Parameter#
IMAGE_WIDTH = 128;
IMAGE_HEIGHT = 128;  
IMAGE_CHANNEL = 3;

#Inference Parameter#
size_in = 0;
size_out= 0;
CLASS_NUM = 2;

#Training Parameter#
LR = 0.001
initial_learning_rate  = 0.0001
BATCH_SIZE =64
CAPACITY = 2000
if BABYTRAIN:
    MAX_STEP = 3000
else:
    MAX_STEP = 1

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 3
CONV2_KENEL_NUM = 40
CONV2_KENEL_SIZE = 3
CONV3_KENEL_NUM = 60
CONV3_KENEL_SIZE = 3
CONV4_KENEL_NUM = 80
CONV4_KENEL_SIZE = 3
CONV5_KENEL_NUM =  100
CONV5_KENEL_SIZE = 3
CONV6_KENEL_NUM =  2
CONV6_KENEL_SIZE = 3
L2_BETA = 0.01





### 1.Creat data ### 
#number 1 to 10 data
train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Dataset1and2\\'
validate_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Dataset3\\'
test_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Dataset3\\'
logs_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_logs\\IR_TrainLogs\\Dataset1and2\\'
model_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_Models\\Dataset1and2\\'

if DELETE == True:
    shutil.rmtree(logs_train_dir,ignore_errors=False, onerror=None)
    shutil.rmtree(model_train_dir,ignore_errors=False, onerror=None)
else:
    pass

train,train_label = input_data.get_files(train_dir)

train_batch, train_label_batch = input_data.get_batch(train,
                                                      train_label,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_CHANNEL,
                                                      BATCH_SIZE, 
                                                      shuffle=True,
                                                      number_thread=1000,
                                                      capacity=CAPACITY) 


validate,validate_label = input_data.get_files(validate_dir)

VALIDATE_BATCH_SIZE = len(validate)
print(VALIDATE_BATCH_SIZE)

validate_batch, validate_label_batch = input_data.get_batch(validate,
                                                            validate_label,
                                                            IMAGE_WIDTH,
                                                            IMAGE_HEIGHT,
                                                            IMAGE_CHANNEL,
                                                            200,
                                                            shuffle=True,
                                                            number_thread=1000,
                                                            capacity=CAPACITY) 

test,test_label = input_data.get_files(test_dir)

TEST_BATCH_SIZE = len(test)
print(TEST_BATCH_SIZE)

test_batch, test_label_batch = input_data.get_batch(test,
                                                    test_label,
                                                    IMAGE_WIDTH,
                                                    IMAGE_HEIGHT,
                                                    IMAGE_CHANNEL,
                                                    TEST_BATCH_SIZE,
                                                    shuffle=False,
                                                    number_thread=1,
                                                    capacity=CAPACITY) 





### 2.Define placeholder for inputs to network ###

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,128,128,3],name='Images') # 不规定有多少个图片’None‘，但是每一个图片都有784个点
    ys = tf.placeholder(tf.float32,[None,2],name='Labels') # 不规定有多少个输出’None‘，但是每个输出都是10个点（0-9） 
    trainphase = tf.placeholder(tf.bool,name='trainphase')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    tf.summary.histogram('image',xs)
    tf.summary.histogram('label',ys)

global_step = tf.Variable(0, trainable=False)


### 3. Setup Network ###

# conv1 layer ##
#BN0 = layer.batch_norm_layer(xs,trainphase,'BN0')

size_in = IMAGE_CHANNEL
size_out = CONV1_KENEL_NUM 
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv1,w1,_,_,b1 = layer.conv_layer(inputs=xs,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,name='conv1')
print('conv1 shape= ', conv1.get_shape())
## pool1 layer ##

#d = conv1.get_shape().as_list()[0]
#drop1 = layer.dropout(conv1,keep_prob=0.5,noise_shape=[64,128,128,1],training=trainphase,name='drop1')

#BN1 = layer.batch_norm_layer(conv1,trainphase,'BN1')
#drop1 = layer.dropout(conv1,keep_prob=keep_prob,name='drop1')

Act1 = layer.relu(conv1,name='Tanh1')


 


size_in = CONV1_KENEL_NUM
size_out= CONV1_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool1,_,_ = layer.max_pool_2x2(inputs=Act1,name='maxpool1')
print('pool1 shape= ', pool1.get_shape())


## conv2 layer ##
size_in = CONV1_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv2,w2,_,_,b2 = layer.conv_layer(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,name='conv2')
print('conv2 shape= ', conv2.get_shape())



#BN2 = layer.batch_norm_layer(conv2,trainphase,'BN2')
#drop2 = layer.dropout(conv2,keep_prob=keep_prob,name='drop2')
Act2 = layer.relu(conv2,name='Tanh2')


#
#BN2 = layer.batch_norm_layer(drop2,trainphase,'BN2')
## pool2 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool2,_,_ = layer.max_pool_2x2(inputs=Act2,name='maxpool2')
print('pool2 shape= ', pool2.get_shape())


## conv3 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv3,w3,_,_,b3 = layer.conv_layer(inputs=pool2,size_in=size_in,size_out=size_out,kernel_size=CONV3_KENEL_SIZE,name='conv3')
print('conv3 shape= ', conv3.get_shape())

#
#BN3 = layer.batch_norm_layer(conv3,trainphase,'BN3')
#drop3 = layer.dropout(conv3,keep_prob=keep_prob,name='drop3')
Act3 = layer.relu(conv3,name='Tanh3')

#drop3 = layer.dropoff(conv3,keep_prob=keep_prob,name='drop3')

#BN3 = layer.batch_norm_layer(drop3,trainphase,'BN3')

## pool3 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool3,_,_ = layer.max_pool_2x2(inputs=Act3,name='maxpool3')
print('pool3 shape= ', pool3.get_shape())

## conv4 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV4_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv4,w4,_,_,b4 = layer.conv_layer(inputs=pool3,size_in=size_in,size_out=size_out,kernel_size=CONV4_KENEL_SIZE,name='conv4')
print('conv4 shape= ', conv4.get_shape())

#BN4 = layer.batch_norm_layer(conv4,trainphase,'BN4')
Act4 = layer.relu(conv4,name='Tanh4')

## pool4 layer ##
size_in = CONV4_KENEL_NUM
size_out= CONV4_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool4,_,_ = layer.max_pool_2x2(inputs=Act4,name='maxpool4')
print('pool4 shape= ', pool4.get_shape())

## conv5 layer ##
size_in = CONV4_KENEL_NUM
size_out= CONV5_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv5,w5,_,_,b5 = layer.conv_layer(inputs=pool4,size_in=size_in,size_out=size_out,kernel_size=CONV5_KENEL_SIZE,name='conv5')
print('conv4 shape= ', conv5.get_shape())

#BN4 = layer.batch_norm_layer(conv4,trainphase,'BN4')
Act5 = layer.relu(conv5,name='Tanh5')


## pool5 layer ##
size_in = CONV5_KENEL_NUM
size_out= CONV5_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool5,_,_ = layer.max_pool_2x2(inputs=Act5,name='maxpool5')
print('pool5 shape= ', pool5.get_shape())

## conv6 layer ##
size_in = CONV5_KENEL_NUM
size_out= CONV6_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv6,w6,_,_,b6 = layer.conv_layer(inputs=pool5,size_in=size_in,size_out=size_out,kernel_size=CONV6_KENEL_SIZE,name='conv6')
print('conv6 shape= ', conv6.get_shape())

#BN4 = layer.batch_norm_layer(conv4,trainphase,'BN4')
Act6 = layer.relu(conv6,name='Tanh6')



height = 4
with tf.name_scope('GAP'):
    glpooling = tf.nn.avg_pool(Act6,ksize=[1,height,height,1],strides=[1,height,height,1],padding='VALID')
    tf.summary.histogram('GAP',glpooling)
print('GAP shape= ', glpooling.get_shape())

## flat layer ##
size_in = CONV6_KENEL_NUM
size_out= CLASS_NUM
fc2 = tf.reshape(glpooling,shape=[-1,CLASS_NUM])
tf.summary.histogram('prediction',fc2)
print('fc2 shape= ', fc2.get_shape())


#size_in = CONV4_KENEL_NUM
#size_out= tf.to_int32(IMAGE_HEIGHT*IMAGE_WIDTH*size_in)
#flat = layer.flat_layer(inputs=Act4,shape=[-1,size_out],name='flat1')
#print('flat shape= ', flat.get_shape())
#
#
### fc1 layer ##
#size_in = size_out
#size_out= 500
#fc1 = layer.fc_layer(inputs=flat,size_in=size_in,size_out=size_out,name='fc1')
#print('fc1 shape= ', fc1.get_shape())
#
#Act5 = layer.tanh(fc1,name='Tanh5')
#
### Drop1 layer##
#size_in = 500
#size_out= 500
#drop2 = layer.dropout(inputs=Act5,keep_prob=keep_prob,name='drop2')
#print('drop2 shape= ', drop2.get_shape())
#
### fc2 layer ##
#size_in = 500
#size_out= 3
#fc2 = layer.fc_layer(inputs=drop2,size_in=size_in,size_out=size_out,name='fc2')
#print('fc2 shape= ', fc2.get_shape())


with tf.name_scope('Softmax'):
    prediction = tf.nn.softmax(fc2)
    
print('prediction shape= ', prediction.get_shape())

print('labels shape= ', train_label_batch.get_shape())
   

train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


### 4. The error between prediction and real data ###
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=fc2,weights=1))# softmax + cross_entropy for classification   
    loss = cross_entropy + L2_BETA*(tf.nn.l2_loss(w1)+
                                    tf.nn.l2_loss(w2)+
                                    tf.nn.l2_loss(w3)+
                                    tf.nn.l2_loss(w4)+
                                    tf.nn.l2_loss(w5)+
                                    tf.nn.l2_loss(w6))
#    loss = cross_entropy
#    for var in train_var:
#        loss += L2_BETA*tf.nn.l2_loss(var)
#        print(var)
    tf.summary.scalar('loss',loss)


### 5. Training Setting ###

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=500,decay_rate=1)   

add_global = global_step.assign_add(1) 

with tf.control_dependencies([add_global]):  
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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


confusionmatrix = tf.confusion_matrix(tf.argmax(ys,1),tf.argmax(prediction,1),2,name='Confusion_matrix')

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
    

### Training ##
#
with tf.Session() as sess:
    merged = tf.summary.merge_all()
      
    train_writer = tf.summary.FileWriter(logs_train_dir+'/train',sess.graph)  
    validate_writer = tf.summary.FileWriter(logs_train_dir+'/validate') 
    saver  = tf.train.Saver()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_imgs,tra_lbls = sess.run([train_batch, train_label_batch])            
            _, tra_loss, tra_acc = sess.run([train_step,loss,train_acc],feed_dict={xs:tra_imgs,ys:tra_lbls,keep_prob:0.5,trainphase:True})
            
            if step % 50 == 0:
                print('Step %d, learning rate = %s'%(step,sess.run(learning_rate)))
                print('Step %d, train loss = %.4f, train accuracy = %.4f%%' %(step, tra_loss, tra_acc*100.0))
                val_imgs,val_lbls = sess.run([validate_batch, validate_label_batch])
                val_loss,val_acc,val_cm = sess.run([loss,validate_acc,confusionmatrix],feed_dict={xs:val_imgs,ys:val_lbls,keep_prob:0.5,trainphase:False})
                print('Step %d, validation loss = %.4f, validation accuracy = %.4f%%' %(step, val_loss, val_acc*100.0))
                print('The Confusion matrix = \n%s'%(val_cm))                                                                      
                summary_train = sess.run(merged,feed_dict={xs:tra_imgs,ys:tra_lbls,keep_prob:0.5,trainphase:True})
                train_writer.add_summary(summary_train, step)
                summary_val = sess.run(merged,feed_dict={xs:val_imgs,ys:val_lbls,keep_prob:0.5,trainphase:False})
                validate_writer.add_summary(summary_val, step)
#            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_train_dir, 'IR_%s.ckpt'%DATE)
                saver.save(sess, checkpoint_path, global_step=step)
            if  (step + 1) == MAX_STEP:                      
                tst_imgs,tst_lbls = sess.run([test_batch, test_label_batch])
                for i in range(2):
                    start_time = time.time()
                    tst_acc,Pre,tst_cm = sess.run([test_acc,prediction,confusionmatrix],feed_dict={xs:tst_imgs,ys:tst_lbls,keep_prob:0.5,trainphase:False}) 
                    elapsed_time = (time.time() - start_time)/BATCH_SIZE
#                print('Prediction = \n%s\n Label = %s%%' %(Pre, tst_lbls))
                    print('The final test accuracy = %.4f%%' %(tst_acc*100.0))
                    print('The Confusion matrix = \n%s'%(tst_cm))
                    print('The average processing time is %.5f'%elapsed_time)
                               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
              
