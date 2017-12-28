
import numpy
import matplotlib.pyplot as plt
import pandas
import math
import neurallib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from neurallib.NeuralNetwork import NeuralNetworkGenerator
from neurallib.SamplingFunctions import *
from neurallib.datasetManager.dataset import DataSet

numpy.random.seed(7)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def main():
	#load and plot dset
	headers = ["timestamp", "tagid", "x_pos", "y_pos", "heading", "direction", "energy", "speed", "total_distance"]
	print("Reading dataset...")
	dataset = DataSet("./dataset/2013-11-03_tromso_stromsgodset_raw_first.csv", headers, UniformSampling)
	print("Ploting dataset...")
	print(dataset)
	# dataset = dataframe.values
	# dataset = dataset.astype('float32')

	# # normalize the dataset
	# scaler = MinMaxScaler(feature_range=(0, 1))
	# dataset = scaler.fit_transform(dataset)

	# # split into train and test sets
	# train_size = int(len(dataset) * 0.67)
	# test_size = len(dataset) - train_size
	# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# print(len(train), len(test))

	# look_back = 1
	# trainX, trainY = create_dataset(train, look_back)
	# testX, testY = create_dataset(test, look_back)

	return

if __name__ == "__main__":
	main()