from sklearn import neural_network
import numpy as np
import os
import sys

def print_progress():
	sys.stdout.write(".")
	sys.stdout.flush()

class NeuralNet:
	def __init__(self):
		self.features = "" # Read features from files
		# self.training
		# self.test
		# self.hits
		# self.accuracy
		# self.classifier
		# MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)[source]¶

	def train(self):
		# self.classifier.fit(X, y)
		pass

	def predict(self):
		# self.classifier.predict(X)
		# count hits
		pass

if __name__ == "__main__":
	net = NeuralNet()
	net.train()
	net.predict()
