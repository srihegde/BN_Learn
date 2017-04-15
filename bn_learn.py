import numpy as np
import bayesnet as bn
import math
from random import randint, random


# Randomly initialize the Bayesian Network
def randInitBN(bnet):
	nodes = bnet.num_nodes
	for i in xrange(nodes):
		while not bnet.addEdge(randint(0,nodes-1), randint(0,nodes-1)): pass
	return bnet

# Computing BIC score for a given Bayesian Network
def computeBIC(bnet):
	# Temp scoring. Need to change
	return -2*bnet.num_edges

# Randomly pick next element of search space
def pickNextBN(bnet):
	opt = randint(0,2)
	if opt == 0:
		while not bnet.addEdge(randint(0,bnet.num_nodes), randint(0,bnet.num_nodes)): pass
	elif opt == 1:
		while not bnet.reverseEdge(randint(0,bnet.num_nodes), randint(0,bnet.num_nodes)): pass
	else:
		while not bnet.deleteEdge(randint(0,bnet.num_nodes), randint(0,bnet.num_nodes)): pass
	return bnet


# Search the search space using Simulated Annealing
def searchSimAnn(bnet, conv_delta):
	best_bn = bnet
	temp, delta = 1000,1
	while temp > 0:
		t -= delta
		new_bn = pickNextBN(best_bn)
		del_score = computeBIC(new_bn) - computeBIC(best_bn)
		if del_score > 0:
			best_bn = new_bn
		else:
			r = random()
			if r < math.exp(del_score/temp):
				best_bn = new_bn

	return best_bn


if __name__ == '__main__':

	data_nodes = 11		# Need to add data processing
	bnet = bn.BayesNet(data_nodes)
	bnet = randInitBN(bnet)
	bnet.showNet('initialBN.png')

	bnet = searchSimAnn(bnet, 1)
	bnet.showNet('finalBN.png')