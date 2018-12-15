import cv2
import os
import sys
from utils import load_model
import keras.backend as K
import numpy as np

def print_progress():
	sys.stdout.write(".")
	sys.stdout.flush()

class ImageLoader():
	def __init__(self):
		self.im_paths = []
		self.__init_paths__()

		self.im_data = []
		self.__load_images__()

	def __init_paths__(self):
		base_path = "/home/ml/Downloads/DeepLearningFiles/"
		print("Loading paths")
	
		for directory in os.listdir(base_path):
			for file in os.listdir(base_path + directory):
				self.im_paths.append(base_path + directory + "/" + file)
				print_progress()

		print("\n")

	def __load_images__(self):
		print("Loading image data")
		
		for path in self.im_paths:
			img = cv2.imread(path)

			if img.shape[0] < img.shape[1]:
				img = cv2.transpose(img)
				img = cv2.flip(img, flipCode = 1)

			img = cv2.resize(img, (224, 224))
			img = img / 255.
			self.im_data.append(img)
			print_progress()
		
		print("\n")

	def data(self):
		return self.im_data

	def paths(self):
		return self.im_paths

class Network():
	def __init__(self):
		print("Loading network")
		self.net = load_model()
		self.get_features = K.function([self.net.layers[0].input, K.learning_phase()], [self.net.get_layer("flatten_2").output])

	def extract_features(self, images, paths):
		print("Extracting features")
		car_features = []

		for img in images:
			# TODO: Save features to a file with the class name and instance index		
			car_features.append(self.get_features([img[np.newaxis, ...], 0])[0])
			# np.savez_compressed("cat.npz", features)
			print_progress()

def main():
	loader = ImageLoader()	
	Network().extract_features(loader.data(), loader.paths())

if __name__ == "__main__":
    main()
