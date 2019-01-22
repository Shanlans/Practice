

# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf

import xlsxwriter

import numpy as np
import BasicLayerDef as layer
import input_data as input_data
import time
import tensorflow.contrib.slim as slim

workbook = xlsxwriter.Workbook('D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_TestLogs\\Compare.xlsx')
worksheet = workbook.add_worksheet()

bold = workbook.add_format({'bold': 1})
data_format = workbook.add_format({'num_format': '0.000%'})

worksheet.write('A1', 'Predict', bold)
worksheet.write('B1', 'Label', bold)

##Define Parameter##

DATE = '20171103'
DELETE = True

#Input Parameter#
IMAGE_WIDTH = 28;
IMAGE_HEIGHT = 28;  
IMAGE_CHANNEL = 1;

#Inference Parameter#
size_in = 0;
size_out= 0;
CLASS_NUM = 2;

#Training Parameter#
LR = 0.01
keep_prob=1
BATCH_SIZE = 5
CAPACITY = 2000
MAX_STEP = 2

CONV1_KENEL_NUM = 20
CONV1_KENEL_SIZE = 5
CONV2_KENEL_NUM = 50
CONV2_KENEL_SIZE = 5
FC1_KENEL_NUM = 500





### 1.Creat data ### 
#number 1 to 10 data
test_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_testGan\\'
logs_test_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_TestLogs\\'
model_train_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_Models\\'



test,test_label = input_data.get_files(test_dir)

test_batch, test_label_batch = input_data.get_batch(test,
                                                    test_label,
                                                    IMAGE_WIDTH,
                                                    IMAGE_HEIGHT,
                                                    IMAGE_CHANNEL,
                                                    BATCH_SIZE, 
                                                    CAPACITY) 




### 3. Setup Network ###

# conv1 layer ##

size_in = IMAGE_CHANNEL
size_out = CONV1_KENEL_NUM 
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv1,w1 = layer.conv_layer_withoutRelu(inputs=test_batch,size_in=size_in,size_out=size_out,kernel_size=CONV1_KENEL_SIZE,name='conv1')
print('conv1 shape= ', conv1.get_shape())
## pool1 layer ##

size_in = CONV1_KENEL_NUM
size_out= CONV1_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool1 = layer.max_pool_2x2(inputs=conv1,name='maxpool1')
print('pool1 shape= ', pool1.get_shape())

## conv2 layer ##
size_in = CONV1_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT
IMAGE_WIDTH = IMAGE_WIDTH
conv2,_ = layer.conv_layer_withoutRelu(inputs=pool1,size_in=size_in,size_out=size_out,kernel_size=CONV2_KENEL_SIZE,name='conv2')
print('conv2 shape= ', conv2.get_shape())

## pool2 layer ##
size_in = CONV2_KENEL_NUM
size_out= CONV2_KENEL_NUM
IMAGE_HEIGHT = IMAGE_HEIGHT/2
IMAGE_WIDTH = IMAGE_WIDTH/2
pool2 = layer.max_pool_2x2(inputs=conv2,name='maxpool2')
print('pool2 shape= ', pool2.get_shape())


## flat layer ##
size_in = CONV2_KENEL_NUM
size_out= tf.to_int32(IMAGE_HEIGHT*IMAGE_WIDTH*size_in)
flat = layer.flat_layer(inputs=pool2,shape=[-1,size_out],name='flat1')
print('flat shape= ', flat.get_shape())


## fc1 layer ##
size_in = size_out
size_out= FC1_KENEL_NUM
fc1 = layer.fc_layer(inputs=flat,size_in=size_in,size_out=size_out,name='fc1')
print('fc1 shape= ', fc1.get_shape())

## Drop1 layer##
size_in = FC1_KENEL_NUM
size_out= FC1_KENEL_NUM
drop1 = layer.dropoff(inputs=fc1,keep_prob=keep_prob,name='drop1')
print('drop1 shape= ', drop1.get_shape())

## fc2 layer ##
size_in = FC1_KENEL_NUM
size_out= CLASS_NUM
fc2 = layer.fc_layer_withoutRelu(inputs=drop1,size_in=size_in,size_out=size_out,name='fc2')
print('fc2 shape= ', fc2.get_shape())


with tf.name_scope('Softmax'):
    prediction = tf.nn.softmax(fc2)
    
print('prediction shape= ', prediction.get_shape())

   


### 4. The error between prediction and real data ###
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=test_label_batch,logits=fc2,weights=1))# softmax + cross_entropy for classification      
    tf.summary.scalar('cross_entropy',cross_entropy)


### 5. Training Setting ###

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

### Training ##
#
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_test_dir,sess.graph)  
    saver  = tf.train.Saver()
    saver.restore(sess,"D:\pythonworkspace\TensorflowTraining\exercises\Shen\Practice\ACFdata\IR_Models\ACF_20171103.ckpt-999")
#    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            start_time = time.time()
            tst_loss, tst_acc,Pres,Lbls = sess.run([cross_entropy,Test_acc,prediction,test_label_batch])
            elapsed_time = (time.time() - start_time)/BATCH_SIZE
            
            if step % 50 == 0 or (step + 1) == MAX_STEP:     
                P_dict,L_dict = sess.run([Prediction_dict,Label_dict])
                print('Step %d, Prediction = \n%s\n,Label = \n%s\n%%'%(step,P_dict,L_dict))
                print('The average processing time is %.5f'%elapsed_time)
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
                summary_str = sess.run(merged)
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