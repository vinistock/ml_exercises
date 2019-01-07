# Car model classification

This exercises will classify car models based on images.

## Instalation

Images for classification should be under images/ and the model should be under models/.

Activate Python 2.7 environment and install requirements.

```bash
$ source activate py27
$ conda install --file requirements.txt
```

Observation: imgaug cannot be installed via conda install. Use pip install imgaug instead.

## Usage

There are two steps to run this exercise: extract features from the images and then using the extracted features to train a classifier.

Extracted features are saved in the features folder.

```bash
python extract_car_features.py
```

The picked classifier is a neural network. It will read the features from the folder, train and then test, reporting the accuracy.

```bash
python neural_net.py
```
