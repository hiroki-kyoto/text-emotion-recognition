# 1. compute the information gain
# 2. 
import os

cmtdir = 'comments/'
path_pos = 'haopingfenci.txt'
path_neg = 'chapingfenci.txt'

def _main_() :
	cmt_pos = 0 # count of positive comment
	cmt_neg = 0 # count of negative comment
	terms = [] # list of terms found
	trm_pos = [] # count of positive comments cotains term
	trm_neg = [] # count of negtive comments contains term
	terms_cur = [] # terms contained in current comment
	
	f = open(cmtdir + path_pos)
	cmt = f.readline()
	
	while cmt :
		cmt_pos += 1
		cmt = cmt.split('\t');
		# reset current term list
		del terms_cur[:]

		for item in cmt :
			terms_cur.append(item)
			idx = terms_cur.index(item)
			if idx<len(terms_cur)-1 :
				# met replica with a comment
				terms_cur.pop()
			else :
				terms.append(item)
				idx = terms.index(item)
				
				if idx==len(terms)-1 :
					# item newly found
					trm_pos.append(1)
					trm_neg.append(0)
				else :
					trm_pos[idx] += 1
					# have to remove replica
					terms.pop()

		# begin next line
		del cmt
		cmt = f.readline()

	f = open(cmtdir + path_neg)
	cmt = f.readline()
	
	while cmt :
		cmt_neg += 1
		cmt = cmt.split('\t');
		# reset current term list
		del terms_cur[:]

		for item in cmt :
			terms_cur.append(item)
			idx = terms_cur.index(item)
			if idx<len(terms_cur)-1 :
				# met replica with a comment
				terms_cur.pop()
			else :
				terms.append(item)
				idx = terms.index(item)
				
				if idx==len(terms)-1 :
					# item not found
					trm_neg.append(1)
					trm_pos.append(0)
				else :
					trm_neg[idx] += 1
					# have to remove replica
					terms.pop()

		# begin next line
		del cmt
		cmt = f.readline()
	
	# finally to show the statistical data
	print 'postive total: ',  cmt_pos
	print 'negtive total: ', cmt_neg
	print '####################################'
	for i in terms :
		print i
	print 'term of positive : ', trm_pos
	print 'term of negtive : ', trm_neg

_main_();
