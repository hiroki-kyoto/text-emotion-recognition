# construct a BPNN model for training
import numpy as np

# define the nonlinear method sigmoid function
sigmoid = np.vectorize(lambda(x): 1.0/1.0+exp(-x)))
grad_sigmoid = np.vectorize(lambda(x): sigmoid(x)*(1-sigmoid(x)))

class BPNN(object) :
	"""Back Propagation Neural Network 
	Attributes:
		layers = [
			input, 
			hidden[, hidden2, hidden3, ...] 
			output
			]
		learning_rate = 0.5 by default
	Methods:
		void rand_train(data, feedback, times) - data should have
			dimension as input layer
			feedback should have the same dimension
			of output layer, [times] defines samples of data will 
			be trained
		list apply(data) - apply model to data and
			predict the expected output
	"""

	def __init__(self, layers, learning_rate) :
		self.layers = layers;
		self.learning_rate = learning_rate;
		
		# constructing the network
		if len(self.layers)<3 :
			print 'BPNN Layer setting error! At least 3 layers!'
			return
		self.inputLayerDim = self.layers[0]
		self.outputLayerDim = self.layers[len(layers)-1]
		self.hiddenLayerDim = self.layers[1:(len(layers)-1)]
		
		# allocate memory for connections
		# input-hidden connections
		self.ihc = np.arange(self.inputLayerDim)
		for i in range(len(self.ihc)) :
			self.ihc[i] = np.random.ranf(self.hiddenLayerDim[0])
		# there could be situation if ihconn allocation failed.
		
		# hidden-output-connections
		self.hoc = np.arange(\
		self.hiddenLayerDim[len(self.hiddenLayerDim)-1])
		for i in range(len(self.hoc)) :
			self.hoc[i] = np.random.ranf(self.outputLayerDim)
		
		# connections between k to k+1 hidden layer
		self.hhc = []
		if len(self.hiddenLayerDim)>1 :
			self.hhc = np.arange(len(self.hiddenLayerDim)-1)
			for i in range(len(self.hiddenLayerDim)-1) :
				self.hhc[i] = np.arange(self.hiddenLayerDim[i])
				for j in range(len(self.hhc[i])) :
					self.hhc[i][j] = \
					np.random.ranf(self.hiddenLayerDim[i+1])
		
		# allocation for input and output for each layer
		self.x = np.arange(len(self.layers)-1)
		self.y = np.arange(len(self.outputLayerDim))
		self.e = np.arange(len(self.outputLayerDim))
		for i in range(len(self.x)) :
			self.x[i] = np.arange(len(self.layers[i]))
		# over!
	
	def rand_train(data, feedback, times) :
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
		train_order = np.random.randint(len(data), times)
		for i in train_order :
			del self.x[0]
			self.x[0] = data[train_order[i]]




