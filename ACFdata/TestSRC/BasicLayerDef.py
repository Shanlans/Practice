# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf
import numpy as np

def conv_layer(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.relu(conv+b)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
    return act,w

def conv_layer_withoutRelu(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,conv)
    return conv,w

def conv_layer_withsigm(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.sigmoid(conv+b)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
    return act,w

def conv_layer_withleakyrelu(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.leaky_relu(conv+b)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
    return act,w

def conv_layer_withtah(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.1),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.tanh(conv+b)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
    return act,w

def fc_layer(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        act = tf.nn.relu(tf.matmul(inputs,w)+b)       
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
        return act
    
    
def fc_layer_withoutRelu(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        fc = tf.matmul(inputs,w)+b       
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,fc)
        return fc,w

def max_pool_2x2(inputs,name='maxpool2x2'):
    # stride = [1, X_movement, Y_movement, 1]
    # Must have stride[0]=[3]=1
    # ksize: kernal size, E.g. max pooling will pick up one max value in 2x2 block,so it is kind of filter  
    with tf.name_scope(name):
        max_pool = tf.nn.max_pool(value=inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='mp2x2')
    return max_pool

def dropoff(inputs,keep_prob=1,name='dropoff'):
    with tf.name_scope(name):
        dropoff = tf.nn.dropout(x=inputs,keep_prob=keep_prob)
    return dropoff


def flat_layer(inputs,shape,name='flat'):
    with tf.name_scope(name):
        flat = tf.reshape(tensor=inputs,shape=shape)
    return flat

def batch_norm_layer(inputs, train_phase, name='BN'):
    with tf.name_scope(name):
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
    return normed       