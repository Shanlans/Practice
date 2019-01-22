# -*- coding: utf-8 -*-
# Created by Shanlan Shen on 8/10/2017

import tensorflow as tf

def conv_layer(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.001),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.001,shape=[size_out]),name='Bias')
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
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')+b
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

def conv_layer_withtah(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.001),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.001,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.tanh(conv+b)
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
    return act,w

def conv_layer_withleakyrelu(inputs,size_in,size_out,kernel_size,name="conv"):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,size_in,size_out],stddev=0.001),name='Weight')
#        w = tf.Variable(tf.random_uniform([5,5,size_in,size_out],minval=-1,maxval=1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        conv = tf.nn.conv2d(inputs,w,strides=[1,1,1,1],padding='SAME')
        act = tf.nn.leaky_relu(conv+b,alpha=0.01)
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

def fc_layer_withleakyrelu(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.01),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        act = tf.nn.leaky_relu(tf.matmul(inputs,w)+b,alpha=0.01)       
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
        return act
    
def fc_layer_tahn(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.01),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        act = tf.nn.tanh(tf.matmul(inputs,w)+b)       
        tf.summary.histogram(weights_name,w)
        tf.summary.histogram(biases_name,b)
        tf.summary.histogram(act_name,act)
        return act
    
def fc_layer_sigm(inputs,size_in,size_out,name='fc'):
    with tf.name_scope(name):
        weights_name= '%s_W'%name
        biases_name = '%s_B'%name
        act_name = '%s_Act'%name
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='Weight')
        b = tf.Variable(tf.constant(0.01,shape=[size_out]),name='Bias')
        act = tf.nn.sigmoid(tf.matmul(inputs,w)+b)       
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
        return fc

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

def concat(tensor1,tensor2,name='concat'):
    with tf.name_scope(name):
        concat_temp = tf.concat([tensor1,tensor2],axis=(len(tensor2.get_shape())-1))
        print('          %s layer, Concat after = %s;\n'%(name, concat_temp.get_shape()[-1]))
    return concat_temp
    
        