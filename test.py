# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np

np.random.seed(1)

b = np.linspace(1,36,num=36).reshape((1,6,6,1))
a = tf.constant(b,dtype=tf.int32)

c = tf.space_to_batch_nd(a,block_shape=[3,3],paddings=[[0,0],[0,0]])
d = tf.batch_to_space_nd(c,block_shape=[3,3],crops=[[0,0],[0,0]])


with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(c).shape)
    print(sess.run(c))
    print(sess.run(d).shape)
    print(sess.run(d))