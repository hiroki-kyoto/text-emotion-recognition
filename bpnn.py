# construct a BPNN model for training
import numpy as np

# define the nonlinear method sigmoid function
sigmoid = np.vectorize(lambda(x): 1.0/(1.0+np.exp(-x)))
grad_sigmoid = np.vectorize(lambda(x): sigmoid(x)*(1-sigmoid(x)))

class BPNN(object) :
	"""Back Propagation Neural Network 
	Attributes:
		layers = [
			input, 
			hidden[, hidden2, hidden3, ...] 
			output
			]
		eta = 0.2 by default, eta is learning rate

	Methods:
		void rand_train(data, feedback, times) - data should have
			dimension as input layer
			feedback should have the same dimension
			of output layer, [times] defines samples of data will 
			be trained
		list apply(data) - apply model to data and
			predict the expected output
	"""

	def __init__(self, layers, eta) :
		self.layers = layers
		self.eta = eta
		
		# constructing the network
		if len(self.layers)<3 :
			print 'BPNN Layer setting error! At least 3 layers!'
			return
		self.inputLayerDim = self.layers[0]
		self.outputLayerDim = self.layers[len(layers)-1]
		self.hiddenLayerDim = self.layers[1:(len(layers)-1)]
		
		# allocate memory for connections
		self.w = range(len(self.layers)-1)
		for i in range(len(self.w)) :
			self.w[i] = np.random.ranf(\
			size=[self.layers[i], self.layers[i+1]])
			wsum = np.sum(self.w[i])
			self.w[i] = self.w[i]/wsum
		
		# allocation for input and output for each layer
		self.x = range(len(self.layers)) # input and output
		self.d = range(len(self.layers)) # adjustment
		self.e = []
		self.esum = 0.0
		self.time = 0
		# over!
	
	def rand_train(self, data, feedback, times) :
		if len(data[0]) != self.inputLayerDim :
			print 'TRAINING INPUT DIMENSION DOES NOT \
			MATCH WITH INPUT LAYER! PROGRAM EXIT WITH ERROR!'
			return
		if len(feedback[0]) != self.outputLayerDim :
			print 'TRAINING FEEDBACK DIMENSION DOES NOT \
			MATCH WITH OUTPUT LAYER! PROGRAM EXIT WITH ERROR!'
		if len(feedback) != len(data) :
			print 'FEEDBACK DIMENSION DOES NOT MATCH WITH INPUT!\
			PROGRAM EXIT WITH ERROR!'
		train_order = np.random.randint(len(data), size=times)
		self.x[0] = np.zeros([1,self.layers[0]])

		for i in train_order :
			# input
			self.x[0][0] = data[i]
			# unifying the input
			xsum = np.sum(self.x[0])
			self.x[0] = self.x[0]/xsum
			
			# forward computing
			for k in range(1, len(self.x)) :
				self.x[k] = np.zeros([1, self.layers[k]])
				self.x[k][0] = sigmoid(self.x[k-1].dot(self.w[k-1]))

			# compute error
			y = np.zeros([1,len(feedback[0])])
			y[0] = feedback[i]
			self.e = self.x[len(self.x)-1] - y
			self.esum += np.sqrt(np.sum(self.e*self.e))
			self.time += 1
			print 'mean, error: ', self.esum/self.time

			# back propagation
			# get last layer adjustment
			self.d[len(self.d)-1] = self.x[len(self.x)-1]*\
			(1-self.x[len(self.x)-1])*\
			(self.x[len(self.x)-1]-y)
			self.w[len(self.w)-1] = \
			self.w[len(self.w)-1] - self.eta*\
			np.dot(np.array(zip(*(self.x[len(self.x)-2]))), \
			self.d[len(self.d)-1])

			# get each layer adjustment
			k = len(self.d)-2
			while k > 0 :
				self.d[k] = self.x[k]*(1-self.x[k])*\
				np.sum(self.w[k]*self.d[k+1], axis=1)
				# adjust weight matrix of each layer
				self.w[k-1] = self.w[k-1] - self.eta*\
				np.dot(np.array(zip(*(self.x[k-1]))), self.d[k])
				k = k-1

		# TRAINING OVER
		print 'TRAINING OVER'
	
	def apply(self, data) :
		# prepare for returned result
		ret = np.zeros([len(data), self.outputLayerDim])

		if len(data[0]) != self.inputLayerDim :
			print 'TRAINING INPUT DIMENSION DOES NOT \
			MATCH WITH INPUT LAYER! PROGRAM EXIT WITH ERROR!'
			return

		x[0] = np.zeros([1, self.layers[0]])
		for i in range(len(data)) :
			self.x[0][0] = np.array(data[i])
			
			# forward computing
			for k in range(1, len(self.x)) :
				self.x[k] = np.zeros([1, self.layers[k]])
				self.x[k][0][:] = sigmoid(self.x[k-1].dot(self.w[k-1]))
			ret[i][:] = self.x[len(self.x)-1]

		return ret

