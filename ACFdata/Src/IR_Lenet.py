# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import os
import shutil
import time

import numpy as np
import BasicLayerDef as layer
import input_data_ir_3class as input_data

import tensorflow.contrib.slim as slim
##Define Parameter##

DATE = '20171013'
DELETE = True

#Input Parameter#
IMAGE_WIDTH = 28;
IMAGE_HEIGHT = 28;  
IMAGE_CHANNEL = 1;

#Inference Parameter#
size_in = 0;
size_out= 0;
CLASS_NUM = 3;

#Training Parameter#
LR = 0.0001
keep_prob=0.3
BATCH_SIZE = 64
VALIDATE_BATCH_SIZE = 160
TEST_BATCH_SIZE = 90
CAPACITY = 2000
MAX_STEP = 5000

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 5
CONV2_KENEL_NUM = 50
CONV2_KENEL_SIZE = 5
FC1_KENEL_NUM = 1024





### 1.Creat data ### 
#number 1 to 10 data
train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Validate\\IR_validate7\\'
validate_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Train\\IR_train7\\'
test_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\Validate\\IR_validate8\\'
logs_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_logs\\IR_TrainLogs\\IR_TAN_4CONV_GAP_2BN_dropoff_3class\\'
model_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_Models\\IR_TAN_4CONV_GAP_2BN_dropoff_3class\\'

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
                                                      CAPACITY) 


validate,validate_label = input_data.get_files(validate_dir)

validate_batch, validate_label_batch = input_data.get_batch(validate,
                                                            validate_label,
                                                            IMAGE_WIDTH,
                                                            IMAGE_HEIGHT,
                                                            IMAGE_CHANNEL,
                                                            VALIDATE_BATCH_SIZE, 
                                                            CAPACITY) 

test,test_label = input_data.get_files(test_dir)

test_batch, test_label_batch = input_data.get_batch(test,
                                                    test_label,
                                                    IMAGE_WIDTH,
                                                    IMAGE_HEIGHT,
                                                    IMAGE_CHANNEL,
                                                    TEST_BATCH_SIZE, 
                                                    CAPACITY) 





### 2.Define placeholder for inputs to network ###

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,28,28,1],name='Images') # 不规定有多少个图片’None‘，但是每一个图片都有784个点
    ys = tf.placeholder(tf.float32,[None,3],name='Labels') # 不规定有多少个输出’None‘，但是每个输出都是10个点（0-9） 


### 3. Setup Network ###

# conv1 layer ##

size_in = IMAGE_CHANNEL
size_out = CONV1_KENEL_NUM 
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv1,_,_,_,_ = layer.conv_layer(inputs=xs,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,name='conv1')
print('conv1 shape= ', conv1.get_shape())

act1 = layer.relu(conv1,'r1')
## pool1 layer ##

size_in = CONV1_KENEL_NUM
size_out= CONV1_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool1,_,_ = layer.max_pool_2x2(inputs=act1,name='maxpool1')
print('pool1 shape= ', pool1.get_shape())

## conv2 layer ##
size_in = CONV1_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv2,_,_,_,_  = layer.conv_layer(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,name='conv2')
print('conv2 shape= ', conv2.get_shape())

act2 = layer.relu(conv2,'r2')

## pool2 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool2,_,_ = layer.max_pool_2x2(inputs=act2,name='maxpool2')
print('pool2 shape= ', pool2.get_shape())


## flat layer ##
size_in = CONV2_KENEL_NUM
size_out= tf.to_int32(IMAGE_HEIGHT*IMAGE_WIDTH*size_in)
flat = layer.flat_layer(inputs=pool2,shape=[-1,size_out],name='flat1')
print('flat shape= ', flat.get_shape())


## fc1 layer ##
size_in = size_out
size_out= FC1_KENEL_NUM
fc1,_,_ = layer.fc_layer(inputs=flat,size_in=size_in,size_out=size_out,name='fc1')
print('fc1 shape= ', fc1.get_shape())



## Drop1 layer##
size_in = FC1_KENEL_NUM
size_out= FC1_KENEL_NUM
drop1 = layer.dropout(inputs=fc1,keep_prob=keep_prob,name='drop1')
print('drop1 shape= ', drop1.get_shape())

act3 = layer.relu(drop1,'r3')

## fc2 layer ##
size_in = FC1_KENEL_NUM
size_out= CLASS_NUM
fc2,_,_ = layer.fc_layer(inputs=act3,size_in=size_in,size_out=size_out,name='fc2')
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
            
#            _, tra_loss, tra_acc, Pre, Lable = sess.run([train_step,cross_entropy,train_acc,prediction,train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_step,cross_entropy,train_acc],feed_dict={xs:tra_imgs,ys:tra_lbls})
            
#            sess.run(train_step,feed_dict={xs:tra_imgs,ys:tra_lbls})
            if step % 50 == 0:
                print('Step %d, train loss = %.4f, train accuracy = %.4f%%' %(step, tra_loss, tra_acc*100.0))
                val_imgs,val_lbls = sess.run([validate_batch, validate_label_batch])
                val_loss,val_acc = sess.run([cross_entropy,validate_acc],feed_dict={xs:val_imgs,ys:val_lbls})
                print('Step %d, validation loss = %.4f, validation accuracy = %.4f%%' %(step, val_loss, val_acc*100.0))
#                print('Step %d, Prediction = %s\n Label = %s%%' %(step, Pre, Lable))  
                summary_train = sess.run(merged,feed_dict={xs:tra_imgs,ys:tra_lbls})
                train_writer.add_summary(summary_train, step)
                summary_val = sess.run(merged,feed_dict={xs:val_imgs,ys:val_lbls})
                validate_writer.add_summary(summary_val, step)
#            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(model_train_dir, 'ACF_%s.ckpt'%DATE)
                saver.save(sess, checkpoint_path, global_step=step)
            if  (step + 1) == MAX_STEP:
                tst_imgs,tst_lbls = sess.run([test_batch, test_label_batch])
                start_time = time.time()
                tst_acc,Pre = sess.run([test_acc,prediction],feed_dict={xs:tst_imgs,ys:tst_lbls}) 
                elapsed_time = (time.time() - start_time)/BATCH_SIZE
#                print('Prediction = \n%s\n Label = %s%%' %(Pre, tst_lbls))
                print('The final test accuracy = %.4f%%' %(tst_acc*100.0))
                print('The average processing time is %.5f'%elapsed_time)
                               
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
              
