import numpy as np
import math
import datetime
from random import randint, random

import bayesnet as bn


# Randomly initialize the Bayesian Network
def randInitBN(bnet):
	nodes = bnet.num_nodes
	for i in xrange(nodes):
		while not bnet.addEdge(randint(0,nodes-1), randint(0,nodes-1)): pass
	return bnet

# Randomly pick next element of search space
def pickNextBN(bnet):
	opt = randint(0,2)
	cnt,attempts = 0,20
	if opt == 0:
		while not bnet.addEdge(randint(0,bnet.num_nodes-1), randint(0,bnet.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(bnet)
	elif opt == 1:
		while not bnet.reverseEdge(randint(0,bnet.num_nodes-1), randint(0,bnet.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(bnet)
	else:
		while not bnet.deleteEdge(randint(0,bnet.num_nodes-1), randint(0,bnet.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(bnet)
	return bnet


# Search the search space using Simulated Annealing
def searchSimAnn(bnet, temp = 1000, delta = 1):
	best_bn = bnet
	max_temp = temp
	while temp > 0:

		new_bn = pickNextBN(best_bn)
		del_score = new_bn.getBIC() - best_bn.getBIC()
		
		if del_score > 0:
			best_bn = new_bn
		else:
			r = random()
			if r < math.exp(del_score/temp):
				best_bn = new_bn
		
		if temp%10 == 0:
			print '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+' ---','Iteration ',max_temp-temp,':'
		if(temp%250 == 0):
			best_bn.showNet('interBN_'+str(max_temp-temp)+'.png')
		temp -= delta

	return best_bn


if __name__ == '__main__':

	data_nodes = 11		# Need to add data processing
	labels = ['praf','pmek','plcg','PIP2','PIP3','p44/42','pakts473','PKA','PKC','P38','pjnk']
	bnet = bn.BayesNet(data_nodes, labels)
	bnet = randInitBN(bnet)
	bnet.showNet('initialBN.png')

	bnet = searchSimAnn(bnet, 20)
	bnet.showNet('./learntStructure/finalBN.png')