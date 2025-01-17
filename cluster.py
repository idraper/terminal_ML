from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import my_numpy
import codecs
import ujson
import time
import math
import sys
import os

# get and format all the data

path = './data/'

cluster_data = []

files = os.listdir(path)

for n, name in enumerate(files):
	# print ('{}\t\t{}'.format(n, name))
	
	with open(path+name, 'rb') as file:
		df = pk.load(file)

	data = df.values
	for i in range(len(data)):
		cluster_data.append(data[i][:210])
		cluster_data.append(list(reversed(data[i][210:-7])))

cluster_data = np.array([np.array(xi) for xi in cluster_data])

print (cluster_data.shape)

# plt.rcParams['figure.figsize'] = (16, 9)

n = 3
# Initializing KMeans
kmeans = MiniBatchKMeans(n_clusters=n, verbose=True)
# Fitting with inputs
kmeans = kmeans.fit(cluster_data)
# Predicting the clusters
labels = kmeans.predict(cluster_data)
# Getting the cluster centers
C = kmeans.cluster_centers_

print(labels.shape)


files = [open('clusters/{}.txt'.format(x), 'w') for x in range(n)]
files = [f.close() for f in files]

files = [open('clusters/{}.txt'.format(x), 'a') for x in range(n)]

for t in range(labels.shape[0]):
	lbl = labels[t]
	data = list(reversed(cluster_data[t].tolist()))

	m = ''
	num = 28
	fill = 0
	for i, val in enumerate(data):
		if num > 0 and i > 0 and i % num == 0:
			fill += 3
			m += '\n{: <{fill}}'.format('', fill=fill)
			num -= 1
		m += ' {} '.format(int(data[i]))

	files[lbl].write('{}\n\n'.format(m))

files = [f.close() for f in files]



def e_dist(x1, x2):
	val = 0
	for a, b in zip(x1,x2):
		val += (a-b)**2
	return math.sqrt(val)

def predict(x1, weights):
	vals = {}
	for i, w in enumerate(weights):
		vals[i] = e_dist(x1, w)
	return min(vals, key=vals.get)

with open('clusters/test.json', 'w') as f:
	ujson.dump(C, f)

with open('clusters/test.json', 'r') as f:
	w = ujson.load(f)

for i, lbl in enumerate(labels):
	print (False) if predict(cluster_data[i], w) != lbl else ''
