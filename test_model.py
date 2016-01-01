# test model with trained weights
import numpy as np
from settings import *
from bpnn import BPNN

def _main_():
	print 'TEST...'
	
	# load features
	print 'LOADING FEATURES FROM FILE'
	f = open(file_features)
	cnt = f.read()
	features = cnt.split('\t')
	features.pop()
	del cnt

	# load comments
	print 'LOADING COMMENTS FROM FILE'
	f = open(cmtdir + path_pos)
	cnt = f.read()
	pos_cmts = cnt.split('\n')
	del cnt
	f = open(cmtdir + path_neg)
	cnt = f.read()
	neg_cmts = cnt.split('\n')
	del cnt

	# split comments into two division
	# about 80% for training, 20% for test
	print 'GETTING DATA READY FOR MODEL TESTMONY'
	train_pos_cmts = pos_cmts[0:80*len(pos_cmts)/100]
	train_neg_cmts = neg_cmts[0:80*len(neg_cmts)/100]
	test_pos_cmts = pos_cmts[len(train_pos_cmts):len(pos_cmts)]
	test_neg_cmts = neg_cmts[len(train_neg_cmts):len(neg_cmts)]

	del train_pos_cmts
	del train_neg_cmts

	# convert text data into vecter of features
	data = np.zeros([len(test_pos_cmts)+len(test_neg_cmts),\
	len(features)])
	# expected output
	expout = range(len(data))

	count = 0
	for i in test_pos_cmts :
		terms = i.split('\t')
		for k in range(len(features)) :
			data[count][k] = 0.1
			for term in terms :
				if features[k] == term :
					data[count][k] = 0.9
		expout[count] = 1
		count += 1
	
	for i in test_neg_cmts :
		terms = i.split('\t')
		for k in range(len(features)) :
			data[count][k] = 0.1
			for term in terms :
				if features[k] == term :
					data[count][k] = 0.9
		expout[count] = 0
		count += 1

	print 'OPEN WEIGHT FILE'
	f = open(file_weight,'r')
	cnt = f.read()
	w = cnt.split('\t')
	w.pop()
	del cnt

	idx = 0
	layers = [len(data[0]), len(data[0])/2, 1]
	model = BPNN(layers, 0.3)
	for i in range(len(model.w)) :
		for j in range(len(model.w[i])) :
			for k in range(len(model.w[i][j])) :
				model.w[i][j][k] = float(w[idx])
				idx += 1
	
	# apply the model
	res = model.apply(data)
	relout = range(len(res))
	for i in range(len(res)) :
		if res[i][0]>=0.5 :
			# means it's pos
			relout[i] = 1
		else :
			relout[i] = 0
	
	# print out the compare
	if len(relout) != len(expout) :
		print 'ERROR IN MODEL RUNNING RESULT'
		return
	print 'COMPARE RESULT'
	correct = 0
	for i in range(len(relout)) :
		if relout[i] == expout[i] :
			correct += 1
	
	print 'MODEL CORRECTNESS: '
	print str(100.0*float(correct)/float(len(relout))) + '%'
	print 'DONE.'

_main_()
