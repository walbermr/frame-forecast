import matplotlib.pyplot as plt

from neurallib.NeuralNetwork import NeuralNetworkGenerator
from neurallib.SamplingFunctions import *
from neurallib.datasetManager.dataset import DataSet

def main():
	dset_path = "./dataset/2013-11-03_tromso_stromsgodset_raw_first.csv"
	headers = ["timestamp", "tagid", "x_pos", "y_pos", "heading", "direction", "energy", "speed", "total_distance"]

	dataset = DataSet(dset_path, headers, type='series', look_back=1)

	nn = NeuralNetworkGenerator("./input/nn.txt", 'lstm', epochs=1, batch_size=1000, scaler=dataset.scalers)
	nn.evaluate(dataset)
	return

if __name__ == "__main__":
	main()
	