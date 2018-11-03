import requests
import numpy as np
import json

def getNumAlgos():
	page = requests.get('http://terminal.c1games.com/leaderboard')
	contents = str(page.content)[2:-1]

	contents = contents[contents.find('algos')+5:]

	search = 'algos" class="value"> '
	pos = contents.find(search)
	return int(contents[pos+len(search):pos+len(search)+5])

	

def searchForAlgoID(algoName):
	for i in range(getNumAlgos(), 0, -1):
		try:
			print ('checking algo {}'.format(i))
			page = requests.get('http://terminal.c1games.com/api/game/algo/{}/matches'.format(8000))
			contents = str(page.content)[2:-1]

			data = json.loads(contents)

			for match in data['matches']:
				w_algo = match['winning_algo']
				l_algo = match['losing_algo']

				w_name = w_algo['name']
				l_name = l_algo['name']
				w_id = w_algo['id']
				l_id = l_algo['id']

				if l_name.upper() == algoName.upper():
					return l_id
				if w_name.upper() == algoName.upper():
					return w_id
		except Exception as e:
			pass

	return -1

def getID(algoName, r=100):
	for i in range(1, r):
		try:
			# print ('checking page {}'.format(i))
			page = requests.get('http://terminal.c1games.com/api/game/leaderboard?page={}'.format(i))
			contents = str(page.content)[2:-1].replace("\\'",'').replace('\\\\','\\').replace('\\"','')
			data = json.loads(contents)

			for algo in data['algos']:
				name = algo['name']
				ID = algo['id']

				if name.upper() == algoName.upper():
					return ID
		except Exception as e:
			print (e)
			break
	return -1

def getLeaderBoardAlgos(pages=[1], limit=1900):
	algos = {}
	for i in pages:
		try:
			# print ('checking page {}'.format(i))
			page = requests.get('http://terminal.c1games.com/api/game/leaderboard?page={}'.format(i))
			contents = str(page.content)[2:-1].replace("\\'",'').replace('\\\\','\\').replace('\\"','')
			data = json.loads(contents)

			for algo in data['algos']:
				name = algo['name']
				ID = algo['id']
				elo = algo['elo']

				if elo > limit:
					algos[name] = ID
				else:
					break
		except Exception as e:
			print (e)
	return algos


def getMatchIDs(algo):
	if type(algo) == str:
		ID = getID(algo)
	elif type(algo) == int:
		ID = algo

	page = requests.get('http://terminal.c1games.com/api/game/algo/{}/matches'.format(ID))
	contents = str(page.content)[2:-1].replace("\\'",'').replace('\\\\','\\').replace('\\"','')
	data = json.loads(contents)

	return [match['id'] for match in data['matches']]

def getMatchStr(mID):
	return 'http://terminal.c1games.com/watch/{}'.format(mID)

def getMatchRawStr(mID):
	return 'http://terminal.c1games.com/api/game/replayexpanded/{}'.format(mID)

def getMatch(mID):
	return str(requests.get(getMatchRawStr(mID)).content)[2:-1].split('\\n')

def getMatchesStr(algo):
	matchIDs = getMatchIDs(algo)
	return [getMatchStr(x) for x in matchIDs]

def getMatchesRawStr(algo):
	matchIDs = getMatchIDs(algo)
	return [getMatchRawStr(x) for x in matchIDs]

def getMatchesStrWithIDs(algo):
	return (getMatchIDs(algo), getMatchesStr(algo))

def getMatchesRawStrWithIDs(algo):
	return (getMatchIDs(algo), getMatchesRawStr(algo))

def getMatchTurnData(mID):
	contents = getMatch(mID)
	mData = {}

	for line in contents:
		line = line.replace("\n", "")
		line = line.replace("\t", "")

		if line != '':
			data = json.loads(line)

			try:
				data['debug']
			except KeyError:
				if data['turnInfo'][2] == 0:
					mData[data['turnInfo'][1]] = data
	return mData

def fact(num):
	if num <= 0: return 0
	return fact(num - 1) + num * 2

def yOffset(y):
	if y < 14:
		return fact(y)
	return fact(14) + (fact(14) - fact(28-y))

def xOffset(x, y):
	if y < 14:
		pass
	elif y == 14:
		y = 13
	else:
		y = 27 - y

	xCount = int(-1 * abs(2 * y - 27) + 28) + 1
	xKey = int (((-1 * 1 / 2) * xCount) + 14)
	return int(x - xKey)


def pointIndex(pos):
	x,y = pos

	yOff = yOffset(y)
	xOff = xOffset(x, y)

	return yOff + xOff

def getFormattedUnitData(mID):
	mData = getMatchTurnData(mID)
	data = {}
	cols =  421	+ 6 	# num of map locs and stats

	for t, info in mData.items():
		filters = []
		destructors = []
		encryptors = []
		pings = []
		emps = []
		scramblers = []
		removes = []

		for i, vals in enumerate(info['p1Units'][:3]):
			if i == 0:
				filters += [(u[0], u[1]) for u in vals]
			if i == 1:
				destructors += [(u[0], u[1]) for u in vals]
			if i == 2:
				encryptors += [(u[0], u[1]) for u in vals]
			if i == 3:
				pings += [(u[0], u[1]) for u in vals]
			if i == 4:
				emps += [(u[0], u[1]) for u in vals]
			if i == 5:
				scramblers += [(u[0], u[1]) for u in vals]
			if i == 6:
				removes += [(u[0], u[1]) for u in vals]

		for i, vals in enumerate(info['p2Units'][:3]):
			if i == 0:
				filters += [(u[0], u[1]) for u in vals]
			if i == 1:
				destructors += [(u[0], u[1]) for u in vals]
			if i == 2:
				encryptors += [(u[0], u[1]) for u in vals]
			if i == 3:
				pings += [(u[0], u[1]) for u in vals]
			if i == 4:
				emps += [(u[0], u[1]) for u in vals]
			if i == 5:
				scramblers += [(u[0], u[1]) for u in vals]
			if i == 6:
				removes += [(u[0], u[1]) for u in vals]


		mapData = np.zeros(cols)

		for pos in filters:
			mapData[pointIndex(pos)] = 1
		for pos in destructors:
			mapData[pointIndex(pos)] = 2
		for pos in encryptors:
			mapData[pointIndex(pos)] = 3
		for pos in pings:
			mapData[pointIndex(pos)] = 4
		for pos in emps:
			mapData[pointIndex(pos)] = 5
		for pos in scramblers:
			mapData[pointIndex(pos)] = 6
		for pos in removes:
			mapData[pointIndex(pos)] = 7

		mapData[421 + 0] = info['p1Stats'][0]
		mapData[421 + 1] = info['p1Stats'][1]
		mapData[421 + 2] = info['p1Stats'][2]
		mapData[421 + 3] = info['p2Stats'][0]
		mapData[421 + 4] = info['p2Stats'][1]
		mapData[421 + 5] = info['p2Stats'][2]

		data[t] = mapData

	return data

def getMatchesData(algo):
	allMatches = {}
	ids = getMatchIDs(algo)

	for ID in ids:
		allMatches[algo, ID] = getMatchTurnData(ID)

	return allMatches

def getMatchesFormatted(algo):
	ids = getMatchIDs(algo)
	allMatches = {}

	for i, ID in enumerate(ids):
		allMatches[(algo, ID)] = getFormattedUnitData(ID)

	return allMatches
