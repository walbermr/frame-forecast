from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping
from .Util import *

from sklearn.metrics import roc_curve, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
import math

class NeuralNetworkGenerator:

	architectures = []
	architectures_scores = []
	__architetures_types = {}

	def __init__(self, path, architecture, epochs = 150, batch_size = 64, look_back=1, scaler=None):
		self.path = path
		self.epochs = epochs
		self.batch_size =  batch_size
		self.architecture = architecture
		self.__look_back = look_back
		self.__scaler = scaler
		self.__architectures_types = {'mlp': self.mlp, 'lstm':self.lstm}

		with open(path, "r") as f:
			qtd_archs = f.readline()
			for _ in range(0, int(qtd_archs)):
				activation, regularization = f.readline().split(" ")
				activation = None if(activation == 'None') else activation.replace('\n', '')
				layers = f.readline().split(" ")
				self.architectures.append({'regularization' : float(regularization),
										   'activation' : activation,
				                           'layers' : list(map(int, layers))
				                           })

	def mlp(self, arch):
		model = Sequential()
		for i in range(len(arch['layers'])):
				print(arch['layers'][i+1])
				if i == 0:
					model.add(Dense(arch['layers'][i+1], input_dim = arch['layers'][i], activation = arch['activation']))
					i += 1

				elif i < len(arch['layers']) - 1:
					model.add(Dense(arch['layers'][i], activation=arch['activation'], activity_regularizer=l2(arch['regularization'])))

				else:
					model.add(Dense(arch['layers'][i], activation = 'sigmoid'))
		return model

	def lstm(self, arch):
		model = Sequential()
		for i in range(len(arch['layers'])):
			if i == 0:
				model.add(LSTM(arch['layers'][i], input_shape=(1, self.__look_back)))
				i += 1

			elif i < len(arch['layers']) - 1:
				model.add(Dense(arch['layers'][i], activation=arch['activation'], activity_regularizer=l2(arch['regularization'])))

			else:
				model.add(Dense(arch['layers'][i]))
		return model


	def evaluate(self, dataset):
		x = 1
		np.random.seed(7)

		for arch in self.architectures:
			model = self.__architectures_types[self.architecture](arch)
			model.summary()

			# Compile model.
			#model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
			model.compile(loss = 'mean_squared_error', optimizer = 'adam')

			# Get dataset variables.
			dset = dataset.get_dataset()

			# Fit the model.
			history = model.fit(dset['X_train'], dset['y_train'], epochs = self.epochs, batch_size = self.batch_size, callbacks = [EarlyStopping(patience = 20)], validation_data = (dset['X_val'], dset['y_val']))

			# Evaluate the model.
			scores = model.evaluate(dset['X_test'], dset['y_test'])
			#self.architectures_scores.append((scores, len(history.history['loss'])))

			y_pred = model.predict(dset['X_test'])

			plot_training_error_curves(history)

			if(self.architecture == 'mlp'):
				y_pred_class = model.predict_classes(dset['X_test'], verbose = 0)
				losses = extract_final_losses(history)
				plot_roc_curve(dset['y_test'], y_pred)
				print('Confusion matrix:')
				print(confusion_matrix(dset['y_test'], y_pred_class))
				print("{metric:<18}{value:.4f}".format(metric = "Train Loss:", value = losses['train_loss']))
				print("{metric:<18}{value:.4f}".format(metric = "Validation Loss:", value = losses['val_loss']))
				print("{metric:<18}{value:.4f}".format(metric = "Test Loss:", value = scores[0]))
				print("{metric:<18}{value:.4f}".format(metric = "Accuracy:", value = accuracy_score(dset['y_test'], y_pred_class)))
				print("{metric:<18}{value:.4f}".format(metric = "Recall:", value = recall_score(dset['y_test'], y_pred_class)))
				print("{metric:<18}{value:.4f}".format(metric = "Precision:", value = precision_score(dset['y_test'], y_pred_class)))
				print("{metric:<18}{value:.4f}".format(metric = "F1:", value = f1_score(dset['y_test'], y_pred_class)))
				print("{metric:<18}{value:.4f}".format(metric = "AUROC:", value = roc_auc_score(dset['y_test'], y_pred)))
			
			elif(self.architecture == 'lstm'):
				dset_full = np.append(dset['y_train'], np.append(dset['y_val'], dset['y_test']))
				dset_full = np.reshape(dset_full, (dset_full.shape[0], 1))
				# make predictions
				train_predict = model.predict(dset['X_train'])				
				val_predict = model.predict(dset['X_val'])
				test_predict = model.predict(dset['X_test'])
				# invert predictions
				train_predict = self.__scaler.inverse_transform(train_predict)
				y_train = self.__scaler.inverse_transform([dset['y_train']])
				val_predict = self.__scaler.inverse_transform(val_predict)
				y_val = self.__scaler.inverse_transform([dset['y_val']])
				test_predict = self.__scaler.inverse_transform(test_predict)
				y_test = self.__scaler.inverse_transform([dset['y_test']])
				# calculate root mean squared error
				print("{metric:<18}{value:.4f}".format(metric = "Train Score RMSE:", value = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))))
				print("{metric:<18}{value:.4f}".format(metric = "Val Score RMSE:", value = math.sqrt(mean_squared_error(y_val[0], val_predict[:,0]))))
				print("{metric:<18}{value:.4f}".format(metric = "Test Score RMSE:", value =math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))))

				plot_series(dset_full, train_predict, val_predict, test_predict, self.__look_back, self.__scaler)

			x += 1

		#self.store_test_scores()

	def store_test_scores(self):
		i = 1
		print("Storing test scores...")
		with open("results/output.txt", "w") as f:
			for scores in self.architectures_scores:
				f.write("\nArchitecture %d \n%s: %.2f%% (%d)" % (i, "accuracy", scores[0][1] * 100, scores[1]))
				i += 1
