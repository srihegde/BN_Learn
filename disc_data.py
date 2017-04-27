import xlrd as xl
from sklearn.cluster import KMeans
import numpy as np
import csv

def readData(fname, num_nodes):
	file = xl.open_workbook(fname)
	sheet = file.sheet_by_index(0)
	cols = []
	print 'Reading Data...'
	for i in xrange(num_nodes):
		cols.append(sheet.col_values(i,1))

	return cols

def sortLabels(data, labs):
	tlabs = []
	tlabs.append([i for i, x in enumerate(labs) if x == 0])
	tlabs.append([i for i, x in enumerate(labs) if x == 1])
	tlabs.append([i for i, x in enumerate(labs) if x == 2])
	low = [data[i] for i in tlabs[0]]
	med = [data[i] for i in tlabs[1]]
	hig = [data[i] for i in tlabs[2]]

	lavg = float(sum(low))/len(low)
	mavg = float(sum(med))/len(med)
	havg = float(sum(hig))/len(hig)

	sar = [(lavg,0),(mavg,1),(havg,2)]
	sar.sort()

	new_tlabs=[]
	for i in xrange(len(sar)):
		new_tlabs.append(tlabs[sar[i][1]])

	new_labs = [-1 for i in xrange(len(data))]
	for i in xrange(3):
		for j in xrange(len(new_tlabs[i])):
			new_labs[new_tlabs[i][j]] = i

	return new_labs


def discretize(data):
	new_data = []
	for i in xrange(len(data)):
		kmeans = KMeans(n_clusters=3, random_state=0).fit([[data[i][j],1] for j in xrange(len(data[i]))])
		labs = kmeans.labels_.tolist()
		labs = sortLabels(data[i],labs)
		new_data.append(labs)
		print labs.count(0), labs.count(1), labs.count(2)

	return new_data

def writeData(fname, data):
	data = np.array(data).transpose().tolist()
	with open(fname, 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(data)


if __name__ == '__main__':
	
	raw_data = readData('./data/1. cd3cd28.xls', 11)
	disc_data = discretize(raw_data)
	writeData('./data/1.csv', disc_data)

	# labs = sortLabels([1,4,2,5,2,10,7,8],[0,2,1,0,0,1,1,2])
	# print labs
	# print disc_data