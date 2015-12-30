# extract features out of list of terms
import os
import math
from settings import *

# some constants
IGmin = 0.001

def _main_() :
	# load features into memory
	f = open(file_terms, 'r')
	cnt = f.read()
	terms = cnt.split('\t')
	terms.pop()
	del cnt
	
	f = open(file_IG, 'r')
	cnt = f.read()
	IGs = cnt.split('\t')
	IGs.pop()
	ids = []
	
	for i in range(0, len(IGs)) :
		IG = float(IGs[i])
		if IG>=IGmin :
			ids.append(i)
	
	print 'Got features: ', str(len(ids))

	# write into file
	f = open(file_features, 'w')
	for i in ids :
		f.write(terms[i] + '\t')
	f.close()

	print 'done.'

	return 0


_main_()
