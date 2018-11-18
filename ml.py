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
def save_model(model):
	keras.models.save_model(
		model,
		'{}/save/model.hdf5'.format(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')),
		overwrite=False,
		include_optimizer=True
	)


# get and format all the data

path = './data/'

train_data = []
train_lbls = []
tmp = []

files = os.listdir(path)
for n, name in enumerate(files):
	with open(path+name, 'rb') as file:
		df = pk.load(file)

	data = df.values
	for i in range(len(data) - 1):
		train_data.append(data[i])
		lbls = data[i+1][:-6]
		tmp.append(lbls[:int(len(lbls)/2)])

for turn in tmp:
	t = []
	for pos in turn:
		if pos == 0:
			t += [0,0,0]
		if pos == 1:
			t += [1,0,0]
		if pos == 2:
			t += [0,1,0]
		if pos == 3:
			t += [0,0,1]
	train_lbls.append(t)

train_data = np.array([np.array(xi) for xi in train_data])
train_lbls = np.array([np.array(xi) for xi in train_lbls])

train_data = normalize(train_data)
# train_lbls = normalize(train_lbls)

# at this point data is fully formatted, now to train the model

print (train_data.shape)
print (train_lbls.shape)

model = keras.Sequential([
	keras.layers.Dense(500, input_shape=(train_data.shape[1],)),
	keras.layers.Dense(600, activation=tf.nn.sigmoid),
	keras.layers.Dense(700, activation=tf.nn.sigmoid),
	keras.layers.Dense(train_lbls.shape[1], activation=tf.nn.sigmoid),
])


# RMSprop lr=.001
# Adadelta lr=2
# Adam lr=.001
# NAdam lr=.005

model.compile(optimizer=keras.optimizers.RMSprop(lr=.001), 
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

tb = keras.callbacks.TensorBoard(log_dir='./ML_summaries/', histogram_freq= 0, write_graph=True, write_images=True)

save = keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

stop = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001)

lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10)

cb_list = [tb, lr]

model.fit(train_data, train_lbls, epochs=5, callbacks=cb_list)
save_model(model)

dim_x = len(model.get_layer(index=0).get_weights()[0])
dim_y = len(model.get_layer(index=0).get_weights()[0][0])







# here and below is all saving/loading the weights

file_path = "weights/model"
indicies = [0,1,2,3]
for i in indicies:
	json.dump(model.get_layer(index=i).get_weights()[0].tolist(), codecs.open('{}_w{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
	json.dump(model.get_layer(index=i).get_weights()[1].tolist(), codecs.open('{}_b{}.json'.format(file_path, i+1), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


weights = []
biases = []

print ("\nLoading...")
start_time = time.time()

for i in indicies:
	with open('{}_w{}.json'.format(file_path, i+1)) as file:
		weights.append(ujson.load(file))
	with open('{}_b{}.json'.format(file_path, i+1)) as file:
		biases.append(ujson.load(file))

elapsed_time = time.time() - start_time
print ('Loading time: {}'.format(elapsed_time))

x = [y%10 for y in range(dim_x)]

def forward_prop(input, indicies, weights, biases):
	o = my_numpy.add_vecs(my_numpy.mult_vec(weights[0],x), biases[0])
	for i in indicies[1:]:
		o = my_numpy.add_vecs(my_numpy.mult_vec(weights[i],o), biases[i])
	o = my_numpy.sigmoid(o)
	return o


start_time = time.time()

o = forward_prop(x, indicies, weights, biases)

elapsed_time = time.time() - start_time
print ('\nMatrix Operations Time: {}'.format(elapsed_time))
