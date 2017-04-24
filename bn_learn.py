import numpy as np
import math
from random import randint, random
import xlrd as xl

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
def searchSimAnn(bnet, conv_delta, temp = 1000, delta = 1):
	best_bn = bnet
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
			print 'Iteration ',1000-temp,':'
		if(temp%250 == 0):
			best_bn.showNet('interBN_'+str(1000-temp)+'.png')
		temp -= delta

	return best_bn


if __name__ == '__main__':

	data_nodes = 11		# Need to add data processing
	bnet = bn.BayesNet(data_nodes)
	bnet = randInitBN(bnet)
	bnet.showNet('initialBN.png')

	bnet = searchSimAnn(bnet, 1, 1000)
	bnet.showNet('finalBN.png')