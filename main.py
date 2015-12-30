# 1. compute the information gain
# 2. extract the feature out
# 3. using neural network to train the model
# 4. test and validation of model 
import os
import math

cmtdir = 'comments/'
path_pos = 'haopingfenci.txt'
path_neg = 'chapingfenci.txt'
file_terms = 'terms.txt'
file_trm_pos = 'trm_pos.txt'
file_trm_neg = 'trm_neg.txt'

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
	
	# save terms into files
	f = open(file_terms, 'w')
	for i in terms :
		f.write(i + '\t')
	f.close()
	f = open(file_trm_pos, 'w')
	for i in trm_pos :
		f.write(str(i) + '\t')
	f.close()
	f = open(file_trm_neg, 'w')
	for i in trm_neg :
		f.write(str(i) + '\t')
	f.close()

	# now calculate the information gain
	cmt_tot = float(cmt_pos + cmt_neg)
	ppos = cmt_pos/cmt_tot
	pneg = 1.0 - ppos
	Hpos = -1.0*ppos*(math.log(ppos,2))
	Hneg = -1.0*pneg*(math.log(pneg,2))
	Htot = float(Hpos + Hneg)

	IGterms = range(0, len(terms))
	for i in range(0, len(terms)) :
		trm_tot = float(trm_pos[i] + trm_neg[i])
		pterm = trm_tot/cmt_tot

		if trm_pos[i]==0 :
			Htrm_pos = 0.0
		else :
			pterm_pos = trm_pos[i]/trm_tot
			Htrm_pos = -1.0*pterm_pos* \
			math.log(pterm_pos)/math.log(2)
		if trm_neg[i]==0 :
			Htrm_neg = 0.0
		else :
			pterm_neg = trm_neg[i]/trm_tot
			Htrm_neg = -1.0*pterm_neg* \
			math.log(pterm_neg)/math.log(2)
		
		ntrm_tot = float(cmt_tot - trm_tot)
		ntrm_pos = float(cmt_pos - trm_pos[i])
		ntrm_neg = float(cmt_neg - trm_neg[i])
		if ntrm_tot-ntrm_pos-ntrm_neg!=0 :
			print 'PROGRAM RUNNING STATUS :\
			ERROR IN YOUR LOGIC!'
			return 1
		
		if ntrm_pos==0 :
			Hntrm_pos = 0.0
		else :
			pnterm_pos = ntrm_pos/ntrm_tot
			Hntrm_pos = -1.0*pnterm_pos* \
			math.log(pnterm_pos)/math.log(2)
		if ntrm_neg==0 :
			Hntrm_neg = 0.0
		else :
			pnterm_neg = ntrm_neg/ntrm_tot
			Hntrm_neg = -1.0*pnterm_neg* \
			math.log(pnterm_neg)/math.log(2)

		# calculate information gain with or without terms[i]
		IGterms[i] = pterm*(Htrm_pos+Htrm_neg) + \
		(1-pterm)*(Hntrm_pos+Hntrm_neg)
		IGterms[i] = Htot - IGterms[i]
		
	# show the IG
	f = open('information_gain.txt', 'w')
	for i in IGterms :
		f.write(str(i) + '\t')
	f.close()

_main_();
