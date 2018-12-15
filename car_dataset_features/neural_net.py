from sklearn import neural_network
from sklearn import model_selection
import numpy as np
import os
import sys

def print_progress():
	sys.stdout.write(".")
	sys.stdout.flush()

class NeuralNet:
	def __init__(self):
		self.y = []
		self.x = self.__load_features__()
		self.x_train = None
		self.y_train = None
		self.x_test = None
		self.y_test = None
		self.hits = 0
		self.accuracy = None
		self.classifier = neural_network.MLPClassifier(hidden_layer_sizes = (100, 100, 50,))

	def __load_features__(self):
		print("Loading features")
		features = []

		for file in os.listdir("./features/"):
			if not file.startswith("."):
				features.append(np.load("./features/" + file)["arr_0"])
				self.y.append(file.split("_")[0])
				print_progress()

		print("\n")
		return features

	def train(self):
		print("Training neural network")
		self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(self.x, self.y, test_size=0.15)
		self.classifier.fit(self.x_train, self.y_train)

	def predict(self):
		print("Predicting")
		predictions = self.classifier.predict(self.x_test)
		i = 0

		for klass in predictions:
			if klass == self.y_test[i]:
				self.hits += 1

			i += 1

		self.accuracy = (100.0 * self.hits) / len(self.x_test)

	def display(self):
		print("\nResults\n")
		print("Classified " + `len(self.x_test)` + " images")
		print("Accuraccy: " + `self.accuracy`)

if __name__ == "__main__":
	net = NeuralNet()
	net.train()
	net.predict()
	net.display()
