import numpy as np
import pydot
import time
import datetime

class BayesNet():
	
	"""Modelling Bayesian Network"""
	def __init__(self, num_nodes):
		self.num_nodes = num_nodes
		self.grph = np.zeros((num_nodes, num_nodes))

	def addEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		self.grph[a][b] = 1
		if self.isDAG() == False or a == b:
			self.grph[a][b] = 0
			# print 'Could not add edge because it\'s no longer a DAG.'
			return False
		return True

	def deleteEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		self.grph[a][b] = 0

	def reverseEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		if self.grph[a][b] == True:
			self.deleteEdge(a,b)
			if self.addEdge(b,a) == False:
				self.addEdge(a,b)
			

	def isEdge(self, a, b):
		assert ((a >= 0 or a < self.num_nodes) or (b >= 0 or b < self.num_nodes)), 'Vertex index out of bounds!'
		if self.grph[a][b] == 1:
			return True
		return False

	def isDAG(self):
		path = set()
		visited = set()

		def visit(vertex):
			if vertex in visited:
				return False
			visited.add(vertex)
			path.add(vertex)
			for i in xrange(self.num_nodes):
				if self.grph[vertex][i] == 1:
					if (i in path and i != vertex) or visit(i):
						return False
			path.remove(vertex)
			return True

		return any(visit(v) for v in xrange(self.num_nodes))

	def showNet(self):
		graph = pydot.Dot(graph_type='digraph')
		
		for i in xrange(self.num_nodes):
			for j in xrange(self.num_nodes):
				if self.grph[i][j] == 1:
					edge = pydot.Edge(str(i+1), str(j+1))
					graph.add_edge(edge)

		fname = 'bn_'+'{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())+'.png'
		graph.write_png(fname)
		print 'Graph saved as '+ fname
		time.sleep(2)



'''if __name__ == '__main__':
	
	bnet = BayesNet(3)
	bnet.addEdge(0,1)
	bnet.addEdge(1,2)
	# bnet.showNet()

	bnet.addEdge(2,0)
	# bnet.showNet()
	
	bnet.addEdge(0,2)
	bnet.showNet()

	bnet.reverseEdge(0,2)
	bnet.showNet() '''
