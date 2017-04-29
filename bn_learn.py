import numpy as np
import math
import copy
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
	tmp_bn = copy.deepcopy(bnet)
	# tmp_bn = bnet
	opt = randint(0,2)
	cnt,attempts = 0,20
	if opt == 0:
		while not tmp_bn.addEdge(randint(0,tmp_bn.num_nodes-1), randint(0,tmp_bn.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(tmp_bn)
	elif opt == 1:
		while not tmp_bn.reverseEdge(randint(0,tmp_bn.num_nodes-1), randint(0,tmp_bn.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(tmp_bn)
	else:
		while not tmp_bn.deleteEdge(randint(0,tmp_bn.num_nodes-1), randint(0,tmp_bn.num_nodes-1)) and cnt < attempts:
			cnt += 1
		if cnt == attempts:
			return pickNextBN(tmp_bn)
	return tmp_bn


# Search the search space using Simulated Annealing
def searchSimAnn(bnet, temp = 1000, delta = 1, snapInterval = 50, printInterval = 10):
	best_bn = bnet
	itr = 0
	while temp > 0:

		itr += 1
		new_bn = pickNextBN(best_bn)
		del_score = new_bn.getBIC() - best_bn.getBIC()
		
		if del_score > 0:
			best_bn = copy.deepcopy(new_bn)
			# best_bn = new_bn
		else:
			r = random()
			if r < math.exp(del_score/temp):
				best_bn = copy.deepcopy(new_bn)
				# best_bn = new_bn
		
		if itr%printInterval == 0:
			print '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+' ---','Iteration ',itr,': Del_BIC = ', math.fabs(del_score),'  New BIC = ', new_bn.getBIC(), '  Best BIC = ', best_bn.getBIC()
		if(itr%snapInterval == 0):
			best_bn.showNet('./interm/interBN_'+str(itr)+'.png')
		temp -= delta

	return best_bn


if __name__ == '__main__':

	data_nodes = 11		# Need to add data processing
	labels = ['praf','pmek','plcg','PIP2','PIP3','p44/42','pakts473','PKA','PKC','P38','pjnk']
	bnet = bn.BayesNet(data_nodes, labels)
	bnet = randInitBN(bnet)
	bnet.showNet('initialBN.png')

	bnet = searchSimAnn(bnet, 100, 0.1,1,1)
	bnet.showNet('./learntStructure/finalBN.png')