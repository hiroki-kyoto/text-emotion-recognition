# train model by using test data and expected output
# to rate BPNN correctness
import numpy as np
from bpnn import BPNN
from settings import *

def _main_() :
	# load features
	f = open(file_features)
	cnt = f.read()
	features = cnt.split('\t')
	features.pop()
	del cnt

	# load comments
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
	train_pos_cmts = pos_cmts[0:80*len(pos_cmts)/100]
	train_neg_cmts = neg_cmts[0:80*len(neg_cmts)/100]
	test_pos_cmts = pos_cmts[len(train_pos_cmts):len(pos_cmts)]
	test_neg_cmts = neg_cmts[len(train_neg_cmts):len(neg_cmts)]

	# convert text data into vecter of features
	data = np.zeros([len(train_pos_cmts)+len(train_neg_cmts),\
	len(features)])
	feedback = np.zeros([len(data),1])

	count = 0
	for i in train_pos_cmts :
		terms = i.split('\t')
		for k in range(len(features)) :
			data[count][k] = 0
			for term in terms :
				if features[k] == term :
					data[count][k] = 1
		feedback[count][0] = 1
		count += 1
	
	for i in train_neg_cmts :
		terms = i.split('\t')
		for k in range(len(features)) :
			data[count][k] = 0
			for term in terms :
				if features[k] == term :
					data[count][k] = 1
		feedback[count][0] = 0
		count += 1

	model = BPNN([len(data[0]),2*len(data[0]),len(data[0]),1],0.3)
	model.rand_train(data, feedback, 10)

	# save weight data into files

_main_()


