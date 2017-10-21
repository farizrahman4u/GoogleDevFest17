import numpy as np



class NeuralNetwork(object):

	'''
	Neural network with 1 hidden layer
	'''
	def __init__(self, input_dim, hidden_dim, output_dim, lr=0.8):
		'''
		Arguments:
			input_dim: Size of the input layer
			hidden_dim: Size of the hidden layer
			output_dim: Size of the output layer
			lr: learning rate
		'''
		self.input_dim = input_dim
		self.hidden_dim	= hidden_dim
		self.output_dim	= output_dim
		self.lr = lr

		# create weight matrices

		# Hidden layer
		self.W1 = np.random.uniform(size=(input_dim, hidden_dim))
		self.b1 = np.zeros((hidden_dim,))

		#Output layer
		self.W2 = np.random.uniform(size=(hidden_dim, output_dim))
		self.b2 = np.zeros((output_dim,))

		self.weights = [self.W1, self.b1,
						self.W2, self.b2]

	def forward(self, x):
		'''
		Arguments:
			x: input data
		'''
		# Compute hidden layer pre-activation
		self.z1 = np.dot(x, self.W1) + self.b1

		# Compute hidden layer activation
		self.H = self.sigmoid(self.z1)

		# Compute output layer pre-activation
		self.z2 = np.dot(self.H, self.W2) + self.b2

		# compute output
		self.y_hat = self.sigmoid(self.z2)

		return self.y_hat

	def sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

	def sigmoid_prime(self, x):
		return x * (1 - x)

	def mse(self, y, y_hat):
		return np.mean(0.5 * (y - y_hat) ** 2)

	def get_gradients(self, x, y):
		'''
		Compute the gradients for the weights W1, b1, W2 and b2
		Arguments:
			x: input data
			y: expected output
		Returns:
			List containing gradients of the weights with respect 
			to the loss:
			[dEdW1, dEdb1, dEdW2, dEdb2]

		'''
		# Do a forward pass to get the output of the neural network
		y_hat = self.forward(x)

		# pre compute common terms
		delta = -(y - y_hat) * self.sigmoid_prime(y_hat)
		H_prime = self.sigmoid_prime(self.H)

		# Compute W2 gradient
		dEdW2 = np.dot(self.H.T, delta)

		# Compute b2 gradient
		dEdb2 = np.sum(delta, axis=0)

		# Compute W1 gradient
		dEdW1 = np.dot(x.T, np.dot(delta, self.W2.T) * H_prime)

		# Compute b1 gradient
		dEdb1 = np.dot(delta, self.W2.T) * H_prime
		dEdb1 = np.sum(dEdb1, axis=0)

		return [dEdW1, dEdb1, dEdW2, dEdb2]

	def train(self, x, y):
		# compute get_gradients
		grads = self.get_gradients(x, y)

		# update weights using gradient descent
		for weight, grad in zip(self.weights, grads):
			weight -= grad * self.lr



# Initialize neural network object
nn = NeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)

# Training data

x = [[3, 6],
     [4, 5],
     [3, 2],
     [6, 4],
     [5, 2],
     [7, 5],
     [2, 5]]

# convert to numpy array
x = np.array(x, dtype=float)

# Normalize
x /= 24

y = [[75], [83], [61], [93], [86], [99], [60]]
y = np.array(y, dtype=float)
y /= 100  # Maximum mark is 100

# Initialize neural network object
nn = NeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1)

for _ in range(100000):
	nn.train(x, y)
