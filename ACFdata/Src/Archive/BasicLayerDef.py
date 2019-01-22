# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import numpy as np
import math

def padding_cal(out_height,stride,in_height,filter_size):
    with tf.name_scope('cal_pad'):
        pad_along_height = (out_height-1)*stride + filter_size - in_height
        pad_top = pad_along_height / 2
    return pad_top

def tanh(inputs,name="tanh"):
    with tf.name_scope(name):
        act_name = '%s_Act'%name
        tanh = tf.nn.tanh(inputs)
        tf.summary.histogram(act_name,tanh)
        return tanh

def relu(inputs,name="relu"):
    with tf.name_scope(name):
        act_name = '%s_Act'%name
        relu = tf.nn.relu(inputs)
        tf.summary.histogram(act_name,relu)
        return relu

def xavier_initial(inputs,size_in,size_out,kernel_size=5,conv=True,uniform=False,name="Xavier"):
    initial = tf.contrib.layers.xavier_initializer(uniform=uniform)
    if conv:
        weight = tf.get_variable(name,shape=[kernel_size,kernel_size,size_in,size_out],initializer=initial)
    else:
        weight = tf.get_variable(name,shape=[size_in,size_out],initializer=initial)
    return weight
    
    
def conv_atrous_withtan(inputs,size_in,size_out,kernel_size,name="dilated_conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        feature_name= '%s_C'%name
#        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.01),name='Weight')+0.01
        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.001,shape=[size_out]),name='Bias')
        conv = tf.nn.atrous_conv2d(inputs,w,rate=2,padding='VALID')+b
        pad = padding_cal(inputs.get_shape().as_list()[1],1,conv.get_shape().as_list()[1],kernel_size)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(feature_name,conv)
        strides=1
    return conv,w,pad,strides,b

def conv_layer(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        feature_name = '%s_C'%name
#        w = xavier_initial(inputs,size_in,size_out,kernel_size=kernel_size,name=weights_name)
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.01),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')+b
        pad = padding_cal(inputs.get_shape().as_list()[1],1,conv.get_shape().as_list()[1],kernel_size)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(feature_name,conv)
        strides=1
    return conv,w,pad,strides,b


def fc_layer(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_fc'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.01),name='Weight')
#        w = xavier_initial(inputs,size_in,size_out,conv=False,name=weights_name)
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        fc = tf.matmul(inputs,w)+b       
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,fc)
        return fc,w,b

def max_pool_2x2(inputs,name='maxpool2x2'):
    # stride = [1, X_movement, Y_movement, 1]
    # Must have stride[0]=[3]=1
    # ksize: kernal size, E.g. max pooling will pick up one max value in 2x2 block,so it is kind of filter  
    with tf.name_scope(name):
        max_pool = tf.nn.max_pool(value=inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='mp2x2')
        tf.summary.histogram(name,max_pool)
    pad = 0
    strides=2
    return max_pool,pad,strides

def dropout(inputs,keep_prob=1,noise_shape=[1,1,1,1],training=False,name='dropout'):
    with tf.name_scope(name):
        dropout = tf.layers.dropout(inputs=inputs,rate=keep_prob,noise_shape=noise_shape,training=training)
        tf.summary.histogram(name,dropout)
    return dropout


def flat_layer(inputs,shape,name='flat'):
    with tf.name_scope(name):
        flat = tf.reshape(tensor=inputs,shape=shape)
    return flat
        

def svmclass(y,ys,w,class_num,batch_size,name='SVM'):
    with tf.name_scope(name):
        regularization_loss = w
        dot =  tf.multiply(y,ys)
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]), 1 - dot));
        svm_loss = regularization_loss + hinge_loss/batch_size;
    return svm_loss

def batch_norm_layer(inputs, train_phase, name='BN'):
    with tf.name_scope(name):
        beta_name= '%s_beta'%name
        gamma_name = '%s_gamma'%name
        normed_name = '%s_BN'%name
        #x.shape[-1]:
        #x.shape[batch,height,width,depth] in CONV or [batch,depth] in FC
        beta = tf.Variable(tf.constant(0.0, shape=[inputs.get_shape()[-1]]), name='beta', dtype=tf.float32,trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[inputs.get_shape()[-1]]), name='gamma',dtype=tf.float32,trainable=True)
        # if len(x.shape)=4 or 2 
        # axises = [0,1,2] or [0]
        #for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], pass axes=[0, 1, 2].
        #for simple batch normalization pass axes=[0] (batch only).
        #其实就是全局做normalization
        print('          Inputs %s doing batch normalization'%inputs.get_shape())
        
        lens = len(inputs.get_shape())-1
        axises = np.arange(len(inputs.get_shape()) - 1)
        if lens == 1:
            axises = [0]
        elif lens == 3:
            axises = [0,1,2]                
        batch_mean, batch_var = tf.nn.moments(inputs, axises, name='moments')
        
        
#        print('     Batch norm mean shape= %s'%batch_mean.get_shape())
#        print('     Batch norm var shape= %s'%batch_var.get_shape())
        #decay = 0.5 本次值和上次值权重都一样
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #滑动窗口来进行加权平均
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        # 通过train_phase(一个布尔表达式，true选择执行接下来第一个，false选择执行后面一个)
        # 这里可以选择执行 正常均值，也可以选择用ExponentialMovingAverage
        mean, var = tf.cond(train_phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        # Beta = 0 ,Gamma = 1,offset = 0.001
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3,name='bn')
        tf.summary.histogram(normed_name,normed)
        tf.summary.histogram(beta_name,beta)
        tf.summary.histogram(gamma_name,gamma)
    return normed


def outFromIn(conv, layerIn):
  n_in = layerIn[0]
  j_in = layerIn[1]
  r_in = layerIn[2]
  start_in = layerIn[3]
  k = conv[0]
  s = conv[1]
  p = conv[2]
  
  n_out = math.floor((n_in - k + 2*p)/s) + 1
  actualP = (n_out-1)*s - n_in + k 
  pR = math.ceil(actualP/2)
  pL = math.floor(actualP/2)
  
  j_out = j_in * s
  r_out = r_in + (k - 1)*j_in
  start_out = start_in + ((k-1)/2 - pL)*j_in
  return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
 


