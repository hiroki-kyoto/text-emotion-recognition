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
		void train(data, feedback) - data should have
			dimension as input layer
			feedback should have the same dimension
			of output layer
		list predict(data) - apply model to data and
			predict the expected output
		void test(data, expected) - use data to rate
			this model, and print out correctness, it's
			sort of validation
	"""

	def __init__(self, layers, learning_rate) :
		self.layers = layers;
		self.learning_rate = learning_rate;
		# constructing the network
		if len(layers)<3 :
			print 'BPNN Layer setting error! At least 3 layers!'
			return
		self.inputLayerDim = layers[0]
		self.outputLayerDim = layers[len(layers)-1]
		self.hiddenLayerDim = layers[1:(len(layers)-1)]
		# allocate memory for connections
		# input-hidden connections
		self.ihc = np.arange(self.inputLayerDim)
		for i in range(len(self.ihc)) :
			self.ihc[i] = np.arange(self.hiddenLayerDim[0])
		# there could be situation if ihconn allocation failed.
		# hidden-output-connections
		self.hoc = np.arange(\
		self.hiddenLayerDim[len(self.hiddenLayerDim)-1])
		for i in range(len(self.hoc)) :
			self.hoc[i] = np.arange(self.outputLayerDim)
		# connections between k to k+1 hidden layer
		self.hhc = []
		if len(self.hiddenLayerDim)>1 :
			self.hhc = np.arange(len(self.hiddenLayerDim)-1)
			for i in range(len(self.hiddenLayerDim)-1) :
				self.hhc[i] = np.arange(self.hiddenLayerDim[i])
				for j in range(len(self.hhc[i])) :
					self.hhc[i][j] = \
					np.arange(self.hiddenLayerDim[i+1])





