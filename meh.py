import svr_lib as svr
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from datetime import datetime

'''
Steps:
	1. Get all top algo's matches - save as json
	2. Format data 

Save format:
Every column is a data type position, every row is a turn
Every game is stored in an new file (dataframe)

1 is filter
2 is destructor
3 is encryptor

Columns:

(map data)
0 ... 420 p1Health p1Cores p1Bits p2Health p2Cores p2Bits

'''

def updateAllData():
	print ('Updating at time:\t{}'.format(datetime.now().time()))

	for algo in svr.getLeaderBoardAlgos([1,2,3,4,5]):
		data = svr.getMatchesFormatted(algo)
		print (data)
		for match in data:
			name, ID = match

			df = pd.DataFrame.from_dict(data[match], orient='index', columns=list(range(421))+['p1Health','p1Cores','p1Bits','p2Health','p2Cores','p2Bits'])

			df.to_pickle('data/{}_{}.pkl'.format(name, ID))
		break

# updateAllData()
