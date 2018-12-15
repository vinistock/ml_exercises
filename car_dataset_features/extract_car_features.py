import cv2
import os
import sys
from utils import load_model
import keras.backend as K
import numpy as np
from imgaug import augmenters as iaa

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
		base_path = "./images/"
		print("Loading paths")
	
		for directory in os.listdir(base_path):
			if not directory.startswith("."):
				for file in os.listdir(base_path + directory):
					if not file.startswith("."):
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

			aug_seq = iaa.Sequential(
            [
                # iaa.ContrastNormalization((0.5, 1.0)),
                # iaa.AddToHueAndSaturation((-5, 5)),
                # iaa.SaltAndPepper(p=0.1),
            ],
            random_order=True)

			img_aug = aug_seq.augment_image(img)

			img_aug = img_aug / 255.
			self.im_data.append(img_aug)
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
		i = 0

		for img in images:
			features = self.get_features([img[np.newaxis, ...], 0])[0]
			
			if "Onix" in paths[i]:
				file_name = "onix_" + `i`
			elif "Ka" in paths[i]:
				file_name = "ka_" + `i`
			elif "HB20" in paths[i]:
				file_name = "hb20_" + `i`
			elif "Sandero" in paths[i]:
				file_name = "sandero_" + `i`
			else:
				file_name = "gol_" + `i`
			
			np.savez_compressed("features/" + file_name + ".npz", features[0])
			i += 1
			print_progress()

if __name__ == "__main__":
	loader = ImageLoader()	
	Network().extract_features(loader.data(), loader.paths())
