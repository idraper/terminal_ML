import tensorflow as tf
import numpy as np
import random

# cmd: > tensorboard --logdir=$PATH

learning_rate = 0.0005
training_epoch = 100000
display_step = 100
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

writer = tf.summary.FileWriter('./', sess.graph)

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
