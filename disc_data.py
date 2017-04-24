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

def discretize(data):
	new_data = []
	for i in xrange(len(data)):
		kmeans = KMeans(n_clusters=3, random_state=0).fit([[data[i][j],1] for j in xrange(len(data[i]))])
		labs = kmeans.labels_.tolist()
		new_data.append(labs)
		print labs.count(0), labs.count(1), labs.count(2)

	return new_data

def writeData(fname, data):
	data = np.array(data).transpose().tolist()
	with open(fname, 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(data)


if __name__ == '__main__':
	
	raw_data = readData('./data/2. cd3cd28icam2.xls', 11)
	disc_data = discretize(raw_data)
	writeData('./data/2.csv', disc_data)
	# print disc_data