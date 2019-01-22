# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import os
import shutil

import xlsxwriter

import numpy as np
import BasicLayerDef as layer
import input_data as input_data
import time

workbook = xlsxwriter.Workbook('D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_TestLogs\\ACF_TAN_4CONV_GAP_BN\\Compare.xlsx')
worksheet = workbook.add_worksheet()

bold = workbook.add_format({'bold': 1})
data_format = workbook.add_format({'num_format': '0.000%'})

worksheet.write('A1', 'Predict', bold)
worksheet.write('B1', 'Label', bold)



##Define Parameter##

DATE = '20171122'
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
LR = 0.01
keep_prob=0.3
BATCH_SIZE = 90
VALIDATE_BATCH_SIZE = 160
TEST_BATCH_SIZE = 90
CAPACITY = 2000
MAX_STEP = 100

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 5
CONV2_KENEL_NUM = 50
CONV2_KENEL_SIZE = 5
CONV3_KENEL_NUM = 70
CONV3_KENEL_SIZE = 5
CONV4_KENEL_NUM =  4
CONV4_KENEL_SIZE = 3







### 1.Creat data ### 
#number 1 to 10 data
train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_test\\'
logs_validation_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\ACF_TestLogs\\ACF_TAN_4CONV_GAP_BN\\'


if DELETE == True:
    shutil.rmtree(logs_validation_dir,ignore_errors=False, onerror=None)
else:
    pass

test,test_label = input_data.get_files(train_dir)

test_batch, test_label_batch = input_data.get_batch(test,
                                                      test_label,
                                                      IMAGE_WIDTH,
                                                      IMAGE_HEIGHT,
                                                      IMAGE_CHANNEL,
                                                      BATCH_SIZE, 
                                                      CAPACITY) 


trainphase = tf.placeholder(tf.bool,name='trainphase')


### 2.Define placeholder for inputs to network ###


### 3. Setup Network ###

# conv1 layer ##
BN0 = layer.batch_norm_layer(test_batch,trainphase,'BN0')

size_in = IMAGE_CHANNEL
size_out = CONV1_KENEL_NUM 
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv1,_ = layer.conv_layer_withtah(inputs=BN0,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,name='conv1')
print('conv1 shape= ', conv1.get_shape())
## pool1 layer ##

drop1 = layer.dropoff(conv1,keep_prob=keep_prob,name='drop1')

BN1 = layer.batch_norm_layer(drop1,trainphase,'BN1')

size_in = CONV1_KENEL_NUM
size_out= CONV1_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool1 = layer.max_pool_2x2(inputs=BN1,name='maxpool1')
print('pool1 shape= ', pool1.get_shape())

## conv2 layer ##
size_in = CONV1_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv2,w2 = layer.conv_layer_withtah(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,name='conv2')
print('conv2 shape= ', conv2.get_shape())

## pool2 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool2 = layer.max_pool_2x2(inputs=conv2,name='maxpool2')
print('pool2 shape= ', pool2.get_shape())


## conv3 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv3,w3 = layer.conv_layer_withtah(inputs=pool2,size_in=size_in,size_out=size_out,kernel_size=CONV3_KENEL_SIZE,name='conv3')
print('conv3 shape= ', conv3.get_shape())

#drop3 = layer.dropoff(conv3,keep_prob=keep_prob,name='drop3')

#BN3 = layer.batch_norm_layer(drop3,trainphase,'BN3')

## pool3 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV3_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool3 = layer.max_pool_2x2(inputs=conv3,name='maxpool3')
print('pool3 shape= ', pool3.get_shape())

## conv4 layer ##
size_in = CONV3_KENEL_NUM
size_out= CONV4_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv4,w4 = layer.conv_layer_withtah(inputs=pool3,size_in=size_in,size_out=size_out,kernel_size=CONV4_KENEL_SIZE,name='conv4')
print('conv4 shape= ', conv4.get_shape())

height = 4
with tf.name_scope('GAP'):
    glpooling = tf.nn.avg_pool(conv4,ksize=[1,height,height,1],strides=[1,height,height,1],padding='VALID')
print('GAP shape= ', glpooling.get_shape())

## flat layer ##
size_in = CONV4_KENEL_NUM
size_out= CLASS_NUM
fc2 = tf.reshape(glpooling,shape=[-1,CLASS_NUM])
print('fc2 shape= ', fc2.get_shape())


with tf.name_scope('Softmax'):
    prediction = tf.nn.softmax(fc2)
    
print('prediction shape= ', prediction.get_shape())

print('labels shape= ', test_label_batch.get_shape())
   


### 4. The error between prediction and real data ###
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=test_label_batch,logits=fc2,weights=1))# softmax + cross_entropy for classification      
    tf.summary.scalar('cross_entropy',cross_entropy)


### 5. Training Setting ###

#with tf.name_scope('train'):
#    train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

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


Prediction_dict = tf.argmax(prediction,1)
Label_dict = tf.argmax(test_label_batch,1)
Test_acc = evaluation(prediction,test_label_batch)
#
#
#
#
#
### Training ##
#
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_validation_dir,sess.graph)  
    saver  = tf.train.Saver()
    saver.restore(sess,"D:\pythonworkspace\TensorflowTraining\exercises\Shen\Practice\ACFdata\ACF_Models\ACF_TAN_4CONV_GAP_BN\ACF_20171122.ckpt-3999")
#    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            
            start_time = time.time()
            tst_loss, tst_acc,Pres,Lbls = sess.run([cross_entropy,Test_acc,prediction,test_label_batch],feed_dict={trainphase:False})
            elapsed_time = (time.time() - start_time)/BATCH_SIZE
            
            if step % 1 == 0 or (step + 1) == MAX_STEP:     
                P_dict,L_dict = sess.run([Prediction_dict,Label_dict],feed_dict={trainphase:False})
                print('Step %d, Prediction = \n%s\n,Label = \n%s\n%%'%(step,P_dict,L_dict))
                print('The average processing time is %.6f'%elapsed_time)
                print('Step %d, Test loss = %.3f, Test accuracy = %.3f%%' %(step, tst_loss, tst_acc*100.0))
                row = 1 
                col = 0 
                for P in P_dict:                     
                     worksheet.write_number(row,col,P)                    
                     row += 1  
                row = 1
                col = 1
                for L in L_dict:
                    worksheet.write_number(row,col,L) 
                    row += 1
                worksheet.write(row, 0, 'Total', bold)
                worksheet.write(row, 2,tst_acc,data_format)
                summary_str = sess.run(merged,feed_dict={trainphase:False})
                train_writer.add_summary(summary_str, step)
            
#            if step % 2000 == 0 or (step + 1) == MAX_STEP:
#                checkpoint_path = os.path.join(model_train_dir, 'ACF_%s.ckpt'%DATE)
#                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    workbook.close()
    coord.join(threads)
    sess.close()
    
    
              
