import numpy as np
import bayesnet as bn
from random import randint


def randInitBN(bnet):
	nodes = bnet.num_nodes
	for i in xrange(nodes):
		while not bnet.addEdge(randint(0,nodes-1), randint(0,nodes-1)):
			pass
	return bnet


if __name__ == '__main__':

	data_nodes = 11		# Need to add data processing
	bnet = bn.BayesNet(data_nodes)
	bnet = randInitBN(bnet)
	bnet.showNet()