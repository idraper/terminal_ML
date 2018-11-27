from sklearn.preprocessing import normalize
from tensorflow import keras
import tensorflow as tf
import pickle as pk
import numpy as np
import my_numpy
import codecs
import json
import ujson
import time
import os

# cmd: > tensorboard --logdir=$PATH

# save the current model
def save_model(model, name='model'):
	keras.models.save_model(
		model,
		'{}/saves/{}.hdf5'.format(os.path.dirname(os.path.abspath(__file__)).replace('\\','/'), name),
		overwrite=False,
		include_optimizer=True
	)


# get and format all the data

path = './data/'

train_data = []
train_lbls = []
tmp = []

files = os.listdir(path)

train_data = np.zeros((len(files), 100, 427))
tmp_lbls = np.zeros((len(files), 100, 427))

for n, name in enumerate(files):
	# print ('{}\t\t{}'.format(n, name))
	
	with open(path+name, 'rb') as file:
		df = pk.load(file)

	data = df.values

	train_data[n] = np.pad(data, pad_width=((0,100-data.shape[0]), (0,0)), mode='constant', constant_values=(0))

	# for i in range(len(data) - 1):
	# 	train_data.append(data[i])
	# 	lbls = data[i+1][:-6]
	# 	tmp.append(lbls[:int(len(lbls)/2)])

	# # pad the data
	# for i in range(len(train_data) % 100, 100):
	# 	train_data.append(np.zeros(427))
	# 	tmp.append([0 for x in range(int(len(lbls)/2))])


# train_data = np.expand_dims(np.array(train_data), axis=1)
# train_lbls = np.array(train_lbls)

tmp_lbls = np.delete(np.roll(train_data, 1), 0, axis=1)
train_data = np.delete(train_data, -1, axis=1)

train_lbls = np.zeros((len(files), tmp_lbls.shape[1], 630))

for i, file in enumerate(tmp_lbls):
	for j, row in enumerate(file):
		for k, pos in enumerate(row):
			if k >= 210: break
			k *= 3
			if pos == 0:
				train_lbls[i][j][k] = 0
				train_lbls[i][j][k+1] = 0
				train_lbls[i][j][k+2] = 0
			elif pos == 1:
				train_lbls[i][j][k] = 1
				train_lbls[i][j][k+1] = 0
				train_lbls[i][j][k+2] = 0
			elif pos == 2:
				train_lbls[i][j][k] = 0
				train_lbls[i][j][k+1] = 1
				train_lbls[i][j][k+2] = 0
			elif pos == 3:
				train_lbls[i][j][k] = 0
				train_lbls[i][j][k+1] = 0
				train_lbls[i][j][k+2] = 1

# train_data = normalize(train_data)
# train_lbls = normalize(train_lbls)

# at this point data is fully formatted, now to train the model

print (train_data.shape)
print (train_lbls.shape)

model = keras.Sequential([
	# keras.layers.Dense(500, input_shape=(train_data.shape[1],)),
	keras.layers.SimpleRNN(500, input_shape=(99,train_data.shape[2]), return_sequences=True),
	# keras.layers.BatchNormalization(),
	keras.layers.Dense(600, activation=tf.nn.sigmoid),
	# keras.layers.BatchNormalization(),
	keras.layers.Dense(700, activation=tf.nn.sigmoid),
	# keras.layers.BatchNormalization(),
	keras.layers.Dense(train_lbls.shape[2], activation=tf.nn.sigmoid),
	# keras.layers.BatchNormalization(),
])
# indicies = [0,2,4,6]
indicies = [0,1,2,3]

from keras.utils.layer_utils import print_summary
print_summary(model)


# RMSprop lr=.001
# Adadelta lr=2
# Adam lr=.001
# NAdam lr=.005

model.compile(optimizer=keras.optimizers.RMSprop(lr=.001), 
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

tb = keras.callbacks.TensorBoard(log_dir='./ML_summaries/', histogram_freq= 0, write_graph=True, write_images=True)

save = keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

stop = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=3)

lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10)

cb_list = [lr]

model.fit(train_data, train_lbls, epochs=100, batch_size=32, callbacks=cb_list)
# save_model(model, 'algoo_v0.0')

dim_x = len(model.get_layer(index=0).get_weights()[0])
dim_y = len(model.get_layer(index=0).get_weights()[0][0])



# here and below is all saving/loading the weights

'''
# Old code for normal feed forward network
file_path = "weights/model"
for i,pos in enumerate(indicies):
	json.dump(model.get_layer(index=pos).get_weights()[0].tolist(), codecs.open('{}_w{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
	json.dump(model.get_layer(index=pos).get_weights()[1].tolist(), codecs.open('{}_b{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
'''

def save_dense(weights, i, pos, k=0, bias=True):
	json.dump(weights[k].tolist(), codecs.open('{}_w{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
	if bias:
		json.dump(weights[-1].tolist(), codecs.open('{}_b{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
	return i+1

def save_rnn(weights, i, pos):
	i = save_dense(weights, i, pos, bias=False)
	i = save_dense(weights, i, pos, k=1)
	return i


# new code for RNN network
file_path = "weights/model"
i = 0
for pos in indicies:
	weights = model.get_layer(index=pos).get_weights()
	if len(weights) == 2:
		i = save_dense(weights, i, pos)
	else:
		i = save_rnn(weights, i, pos)



weights = {}
biases = {}
layers = 5

print ("\nLoading...")
start_time = time.time()

for i in range(layers):
	with open('{}_w{}.json'.format(file_path, i+1)) as file:
		weights[i] = ujson.load(file)
	try:
		with open('{}_b{}.json'.format(file_path, i+1)) as file:
			biases[i] = ujson.load(file)
	except FileNotFoundError: pass

elapsed_time = time.time() - start_time
print ('Loading time: {}'.format(elapsed_time))

x = [y%10 for y in range(dim_x)]

def forward_prop(input, layers, weights, biases):
	o = my_numpy.mult_vec(weights[0], x)
	try: o = my_numpy.add_vecs(o, biases[0])
	except KeyError: pass
	for i in range(layers - 1):
		o = my_numpy.mult_vec(weights[i+1], o)
		try: o = my_numpy.add_vecs(o, biases[i+1])
		except KeyError: pass
		o = my_numpy.sigmoid(o)
	return o

	# o = my_numpy.add_vecs(my_numpy.mult_vec(weights[0],x), biases[0])
	# for i,_ in enumerate(indicies[1:]):
	# 	o = my_numpy.add_vecs(my_numpy.mult_vec(weights[i+1],o), biases[i+1])
	# 	o = my_numpy.sigmoid(o)
	# return o


start_time = time.time()

o = forward_prop(x, layers, weights, biases)

elapsed_time = time.time() - start_time
print ('\nMatrix Operations Time: {}'.format(elapsed_time))
