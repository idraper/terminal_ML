from sklearn.preprocessing import normalize
from tensorflow import keras
import tensorflow as tf
import pickle as pk
import pandas as pd
import numpy as np
import codecs
import json
import os

# cmd: > tensorboard --logdir=$PATH

path = './data/'

train_data = []
train_lbls = []

files = os.listdir(path)
for n, name in enumerate(files):
	with open(path+name, 'rb') as file:
		df = pk.load(file)

	data = df.values
	for i in range(len(data) - 1):
		train_data.append(data[i])
		train_lbls.append(data[i+1])

train_data = np.array([np.array(xi) for xi in train_data])
train_lbls = np.array([np.array(xi) for xi in train_lbls])

train_data = normalize(train_data)
train_lbls = normalize(train_lbls)

print (train_data.shape)
print (train_lbls.shape)

model = keras.Sequential([
	keras.layers.Dense(500, input_shape=(train_data.shape[1],)),
	keras.layers.BatchNormalization(),
	keras.layers.Dense(500, activation=tf.nn.relu),
	keras.layers.BatchNormalization(),
	keras.layers.Dense(train_data.shape[1], activation=tf.nn.softmax),
	keras.layers.BatchNormalization()
])

model.compile(optimizer=keras.optimizers.SGD(lr=1), 
			  loss='mse',
			  metrics=['accuracy', 'mae'])

tb = keras.callbacks.TensorBoard(log_dir='./ML_summaries/', histogram_freq= 0, write_graph=True, write_images=True)

save = keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

stop = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001)

lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

cb_list = [tb, save, lr]

model.fit(train_data, train_lbls, epochs=10, callbacks=cb_list)

# print (model.get_layer(index=0).get_weights()[1])

# file_path = "model"
# json.dump(model.get_layer(index=0).get_weights()[0].tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
