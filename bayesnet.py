import numpy as np
import pydot
from random import randint, random
import time
import datetime

class BayesNet():
	
	"""Modelling Bayesian Network"""
	def __init__(self, num_nodes):
		self.num_nodes = num_nodes
		self.grph = np.zeros((num_nodes, num_nodes))
		self.num_edges = 0
		self.BIC = 0

	def addEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		self.grph[a][b] = 1
		if self.isCyclic() == True or a == b:
			self.grph[a][b] = 0
			# print 'Could not add edge because it\'s no longer a DAG.'
			return False
		self.num_edges += 1;
		return True

	def deleteEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		if self.grph[a][b] == 1:
			self.grph[a][b] = 0
			self.num_edges -= 1
			return True
		return False


	def reverseEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		if self.grph[a][b] == 1:
			self.deleteEdge(a,b)
			if self.addEdge(b,a) == False:
				self.addEdge(a,b)
				return False
			return True
		return False

			

	def isEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		if self.grph[a][b] == 1:
			return True
		return False


	def isCyclic(self):
		visited = [False for i in xrange(self.num_nodes)]
		recstack = [False for i in xrange(self.num_nodes)]

		def isCyclicUtil(v):
			if not visited[v]:
				visited[v] = True
				recstack[v] = True

				for i in xrange(self.num_nodes):
					if self.grph[v][i] == 1:
						if not visited[i] and isCyclicUtil(i):
							return True
						elif recstack[i]:
							return True

			recstack[v] = False
			return False

		return any(isCyclicUtil(v) for v in xrange(self.num_nodes))
		

	def showNet(self, fname = 'bn_'+'{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+'.png'):
		graph = pydot.Dot(graph_type='digraph')
		
		for i in xrange(self.num_nodes):
			graph.add_node(pydot.Node(str(i+1)))

		for i in xrange(self.num_nodes):
			for j in xrange(self.num_nodes):
				if self.grph[i][j] == 1:
					edge = pydot.Edge(str(i+1), str(j+1))
					graph.add_edge(edge)

		graph.write_png(fname)
		print 'Graph saved as '+ fname
		time.sleep(2)


	# Computing BIC score for a given Bayesian Network
	def getBIC(self):
		# Temp scoring. Need to change
		nodes = self.num_nodes
		prior = 2*self.num_edges/float(nodes*(nodes-1))
		lhood = random()
		self.BIC = lhood*prior
		return self.BIC



'''if __name__ == '__main__':
	
	bnet = BayesNet(3)
	bnet.addEdge(0,1)
	bnet.addEdge(1,0)
	bnet.showNet()'''
