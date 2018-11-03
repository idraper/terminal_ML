import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import pickle as pk
import pandas as pd
import os

# cmd: > tensorboard --logdir=$PATH

path = './data/'
 
files = os.listdir(path)
for name in files:
	with open(path+name, 'rb') as file:

		df = pk.load(file)

	break;

features = list(range(421))+['p1Health','p1Cores','p1Bits','p2Health','p2Cores','p2Bits']

training_df = {}

df = df.to_dict('list')

train_data = []
train_lbls = []

for key, val in df.items():
	train_data.append([val[i] for i in range(len(val)-1)])
	train_lbls.append([val[i+1] for i in range(len(val)-1)])


# for i, k in enumerate(list(range(421))+['p1Health','p1Cores','p1Bits','p2Health','p2Cores','p2Bits']):
# 	df



'''
model = keras.Sequential([
	keras.layers.Flatten(input_shape=421+6),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(210, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=5)
'''

'''
learning_rate = 0.05
training_epoch = 1000
display_step = 10
batch_size = 100
ninputs = 1
nhidden = 15
noutputs = 1

# W * x + b
with tf.name_scope('input'):
	x = tf.placeholder(name='x', dtype='float32', shape=[None,ninputs])
	
with tf.name_scope('variables'):
	W1 = tf.Variable(tf.random_normal([ninputs,nhidden]), name='Weight1', dtype='float32')
	W2 = tf.Variable(tf.random_normal([nhidden,noutputs]), name='Weight2', dtype='float32')
	b1 = tf.Variable(tf.zeros([nhidden]), name='Bias1', dtype='float32')
	b2 = tf.Variable(tf.zeros([noutputs]), name='Bias2', dtype='float32')

#output = W * x + b
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
#output = tf.add(tf.matmul(x, W1), b1)
output = tf.add(tf.matmul(hidden_out, W2), b2)
#output = tf.add(tf.matmul(x, W1), b1)

error = tf.subtract(tf.sin(x), output)
mse = tf.reduce_mean(tf.square(error))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)

c = 0
# summaries for tensorboard
tf.summary.scalar("error", mse)

merged = tf.summary.merge_all()

sess = tf.Session()

writer = tf.summary.FileWriter('./ML_summaries', sess.graph)

sess.run(tf.global_variables_initializer())
for epoch in range(training_epoch):
	#avg_cost = 0.
	
	dict = {x : [[random.uniform(0,1) for x in range(ninputs)]]}
	summary, _, c = sess.run([merged, optimizer, mse], feed_dict=dict)
	
	if (epoch + 1) % display_step == 0:
		writer.add_summary(summary, epoch)
		print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
		#print (sess.run(W1, feed_dict=dict))
		#print (sess.run(W2, feed_dict=dict))
		#print ()

writer.close()
sess.close()

'''
