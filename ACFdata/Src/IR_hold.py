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

import diroperation as makeDir

#Working dict#
DATE = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))

TrainingPlan = 'TFilted10'

operationPath = "D:\pythonworkspace\TensorflowTraining\exercises\Shen\Practice\ACFdata\Src"
os.chdir(operationPath)

IR_database = os.path.join(os.path.dirname(operationPath),'IR_database')
IR_logs = os.path.join(os.path.dirname(operationPath),'IR_logs')
IR_model = os.path.join(os.path.dirname(operationPath),'IR_Models')

trainDir = os.path.join(IR_database,'TrainingData','TrainFilted10\\')
validateDir = os.path.join(IR_database,'ValidateData','ValidateFilted6\\')
testDir = os.path.join(IR_database,'TestData','TestFilted6\\')

trainLogs = os.path.join(IR_logs,'IR_Trainlogs',TrainingPlan,DATE)
testLogs = os.path.join(IR_logs,'IR_testlogs',TrainingPlan,DATE)
modelDir = os.path.join(IR_model,TrainingPlan,DATE)
makeDir.mkdir(modelDir)

#loadModelDir = os.path.join(IR_model,TrainingPlan,'2017-12-27-11-42','IR.ckpt-4999')
#loadModelDir = os.path.join(IR_model,TrainingPlan,'2017-12-27-13-02','IR.ckpt-4999')

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
BATCH_SIZE=80
CAPACITY = 2000

MAX_STEP = 100

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 3
CONV2_KENEL_NUM = 40
CONV2_KENEL_SIZE = 3
CONV3_KENEL_NUM = 60
CONV3_KENEL_SIZE = 3
CONV4_KENEL_NUM = 80
CONV4_KENEL_SIZE = 3
CONV5_KENEL_NUM =  40
CONV5_KENEL_SIZE = 1
CONV6_KENEL_NUM =  2
CONV6_KENEL_SIZE = 3
L2_BETA = 0.0025

#### 1.Creat data ### 

print('------------------------')
print('Train Data Set: ' )

train,train_label = input_data.get_files(trainDir)

train_batch, train_label_batch = input_data.get_batch(train,
                                                      train_label,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_CHANNEL,
                                                      BATCH_SIZE, 
                                                      shuffle=True,
                                                      number_thread=1000,
                                                      capacity=CAPACITY) 

print('Validate Data Set: ' )

validate,validate_label = input_data.get_files(validateDir)

VALIDATE_BATCH_SIZE = len(validate)

validate_batch, validate_label_batch = input_data.get_batch(validate,
                                                            validate_label,
                                                            IMAGE_WIDTH,
                                                            IMAGE_HEIGHT,
                                                            IMAGE_CHANNEL,
                                                            VALIDATE_BATCH_SIZE,
                                                            shuffle=False,
                                                            number_thread=1,
                                                            capacity=CAPACITY) 

print('Test Data Set: ' )

test,test_label = input_data.get_files(testDir)

TEST_BATCH_SIZE = len(test)

test_batch, test_label_batch = input_data.get_batch(test,
                                                    test_label,
                                                    IMAGE_WIDTH,
                                                    IMAGE_HEIGHT,
                                                    IMAGE_CHANNEL,
                                                    TEST_BATCH_SIZE,
                                                    shuffle=False,
                                                    number_thread=1,
                                                    capacity=CAPACITY) 
#
#
#
#
#
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
conv1,w1,_,_,b1 = layer.conv_layer(inputs=xs,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,device=1,name='conv1',trainphase=True)
print('conv1 shape= ', conv1.get_shape())
## pool1 layer ##


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
conv2,w2,_,_,b2 = layer.conv_layer(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,device=1,name='conv2',trainphase=True)
print('conv2 shape= ', conv2.get_shape())


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
conv3,w3,_,_,b3 = layer.conv_layer(inputs=pool2,size_in=size_in,size_out=size_out,kernel_size=CONV3_KENEL_SIZE,device=1,name='conv3',trainphase=True)
print('conv3 shape= ', conv3.get_shape())

Act3 = layer.relu(conv3,name='Tanh3')

#drop3 = layer.dropoff(conv3,keep_prob=keep_prob,name='drop3')



## pool3 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool3,_,_ = layer.max_pool_2x2(inputs=Act3,name='maxpool3')
print('pool3 shape= ', pool3.get_shape())

#BN3 = layer.batch_norm_layer(pool3,trainphase,'BN3')



#def dropout():
#    shape1=tf.cast([128,16,16,1],tf.int32)
#    drop = tf.layers.dropout(BN3,0.5,noise_shape=shape1,training=True) 
#    return drop
#
#def nodropout():
#    return BN3
#
#out3 = tf.cond(trainphase,dropout,nodropout)

## conv4 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV4_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv4,w4,_,_,b4 = layer.conv_layer(inputs=pool3,size_in=size_in,size_out=size_out,kernel_size=CONV4_KENEL_SIZE,device=1,name='conv4',trainphase=True)
print('conv4 shape= ', conv4.get_shape())

#BN4 = layer.batch_norm_layer(conv4,trainphase,'BN4')
Act4 = layer.relu(conv4,name='Tanh4')

# pool4 layer ##
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
conv5,w5,_,_,b5 = layer.conv_layer(inputs=pool4,size_in=size_in,size_out=size_out,kernel_size=CONV5_KENEL_SIZE,device=1,name='conv5',trainphase=True)
print('conv4 shape= ', conv5.get_shape())

#BN4 = layer.batch_norm_layer(conv5,trainphase,'BN5')
#Act5 = layer.relu(conv5,name='Tanh5')

#BN5 = layer.batch_norm_layer(conv5,trainphase,'BN5')


## pool5 layer ##
size_in = CONV5_KENEL_NUM
size_out= CONV5_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool5,_,_ = layer.max_pool_2x2(inputs=conv5,name='maxpool5')
print('pool5 shape= ', pool5.get_shape())

## conv6 layer ##
size_in = CONV5_KENEL_NUM
size_out= CONV6_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv6,w6,_,_,b6 = layer.conv_layer(inputs=pool5,size_in=size_in,size_out=size_out,kernel_size=CONV6_KENEL_SIZE,device=1,name='conv6',trainphase=True)
print('conv6 shape= ', conv6.get_shape())

#BN4 = layer.batch_norm_layer(conv4,trainphase,'BN4')
Act6 = layer.relu(conv6,name='Tanh6')



height = 4
with tf.name_scope('GAP'):
    glpooling = tf.nn.avg_pool(Act6,ksize=[1,height,height,1],strides=[1,height,height,1],padding='VALID')
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
    tf.summary.scalar('loss',loss)


### 5. Training Setting ###

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=300,decay_rate=0.9)   

add_global = global_step.assign_add(1) 

with tf.control_dependencies([add_global]):  
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

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
    train_writer = tf.summary.FileWriter(trainLogs+'/train',sess.graph)  
    validate_writer = tf.summary.FileWriter(trainLogs+'/validate',sess.graph) 
    saver  = tf.train.Saver()
    
#    saver.restore(sess,loadModelDir)
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
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(modelDir, 'IR.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            if  (step + 1) == MAX_STEP:                              
                tst_imgs,tst_lbls = sess.run([test_batch, test_label_batch])
                for i in range(2):
                    start_time = time.time()
                    tst_acc,Pre,tst_cm = sess.run([test_acc,prediction,confusionmatrix],feed_dict={xs:tst_imgs,ys:tst_lbls,keep_prob:0.5,trainphase:False}) 
                    elapsed_time = (time.time() - start_time)/BATCH_SIZE
                    print('The final test accuracy = %.4f%%' %(tst_acc*100.0))
                    print('The Confusion matrix = \n%s'%(tst_cm))
                    print('The average processing time is %.5f'%elapsed_time)
                
                               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
              
