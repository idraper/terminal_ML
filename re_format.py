import numpy as np
import pickle as pk
import pandas as pd
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

print (train_data.shape)
# print (len(train_data[0]))
