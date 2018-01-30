import json
import numpy as np
import pandas as pd
from neurallib.datasetManager.dataframes import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from neurallib.Util import plot_scatter_matrix, plot_boxPlot

class DataSet:
	def __init__(self, path, headers, sampling=None, type='table', look_back=1):
		self.path = path
		self.headers = headers

		main = pd.read_csv(self.path, names=headers)

		if(type == 'table'):
			main.drop_duplicates(inplace = True)

			plot_scatter_matrix(main, 'target')
			plot_boxPlot(main)

			df1 = select_target(main, 'target', 0)
			df2 = select_target(main, 'target', 1)

			# Split dataset.
			df1 = self.split_dataframe(df1)
			df2 = self.split_dataframe(df2)

			df1, df2 = sampling(df1, df2)

			self.dataframe = self.concatenate_and_shuffle_dataset(df1, df2)

		elif(type == 'series'):
			selection = self.__get_selection()
			self.dataframe = []
			self.scalers = []

			__min_size = 0

			for tag in selection["tagids"]:
				dataframe_slice = main[main.tagid == tag]
				d_set_aux = []
				for key in selection["keys"]:
					list_to_append = list(dataframe_slice[key].values)
					if __min_size == 0:
						__min_size = len(list_to_append)

					elif len(list_to_append) - __min_size > 0:
						size = len(list_to_append)
						for i in range(size-__min_size):
							list_to_append.pop()

					elif __min_size - len(list_to_append) > 0:
						size = len(list_to_append)
						for i in range(size-__min_size):
							list_to_append = list_to_append.append(0)

					list_to_append = list(map(lambda x: [x], list_to_append))
					d_set_aux.append(list_to_append)
				self.dataframe.append(d_set_aux)

			self.dataframe, self.scalers = self.series_dataset(self.dataframe, look_back)

	def __create_spl_dframe(self, a, b, c, d, e, f):
		return {'X_train': a,
				'y_train': b,
				'X_test': c,
				'y_test': d,
				'X_val': e,
				'y_val': f}
	
	def __get_selection(self):
		with open("./input/dset_select.txt") as f:
			data = json.loads(f.read())
			return data

	def split_dataframe(self, df):
		X = df.iloc[:, :-1].values
		y = df.iloc[:, -1].values

		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size = 1 / 4, random_state = 42, stratify = y)

		X_train, X_val, y_train, y_val = \
			train_test_split(X_train, y_train, test_size = 1 / 3, random_state = 42, stratify = y_train)

		return self.__create_spl_dframe(X_train, y_train, X_test, y_test, X_val, y_val)

	def concatenate_and_shuffle_dataset(self, df1, df2):
		X_train = np.concatenate((df1['X_train'], df2['X_train']), axis = 0)
		y_train = np.concatenate((df1['y_train'], df2['y_train']), axis = 0)
		X_test = np.concatenate((df1['X_test'], df2['X_test']), axis = 0)
		y_test = np.concatenate((df1['y_test'], df2['y_test']), axis = 0)
		X_val = np.concatenate((df1['X_val'], df2['X_val']), axis = 0)
		y_val = np.concatenate((df1['y_val'], df2['y_val']), axis = 0)

		# SCALER
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		X_val = scaler.transform(X_val)

		train = np.c_[X_train, y_train]
		val = np.c_[X_val, y_val]

		for _ in range(0, 30):
			np.random.shuffle(train)
			np.random.shuffle(val)

		X_train = train[:, :-1]
		y_train = train[:, -1]

		X_val = val[:, :-1]
		y_val = val[:, -1]

		return self.__create_spl_dframe(X_train, y_train, X_test, y_test, X_val, y_val)

	def get_dataset(self):
		return self.dataframe

	def __create_lookback_frame(self, data, look_back):
		shape = data.shape
		dataX, dataY = [], []
		if(len(shape) == 2):
			for i in range(len(data)-look_back-1):
				dataX.append(data[i:(i+look_back), 0])
				dataY.append(data[i+look_back, 0])
		else:
			for i in range(shape[0]):
				a, b = self.__create_lookback_frame(data[i], look_back)
				dataX.append(a)
				dataY.append(b)

		return np.array(dataX), np.array(dataY)

	def __normalize_multidimentional_series(self, dataset, shape):
		if(len(shape) == 2):
			#return normalized
			scaler = MinMaxScaler(feature_range=(0, 1))
			dataset = scaler.fit_transform(dataset)
			return np.array(dataset), scaler
		else:
			dataset_normalized = []
			scalers = []
			for i in range(shape[0]):
				ret0, ret1 = self.__normalize_multidimentional_series(dataset[i], shape[1:])
				dataset_normalized.append(ret0)
				scalers.append(ret1)
			return np.array(dataset_normalized), scalers

	def __split_multidimentional_series(self, dataset, interval):
		shape = dataset.shape
		if(len(shape) == 2):
			return np.array(dataset[interval[0]:interval[1]])
		else:
			dset_aux = []
			for i in range(shape[0]):
				dset_aux.append(self.__split_multidimentional_series(dataset[i], interval))
			return np.array(dset_aux)

	def series_dataset(self, dataset, look_back):
		dataset = np.array(dataset)
		scalers = []
		dataset_shape = dataset.shape
		time_steps = dataset_shape[len(dataset_shape)-2:][0]

		#normalize the dataset for each sample and feature
		dataset, scalers = self.__normalize_multidimentional_series(dataset, dataset_shape)

		# split into train and test sets
		train_size = int(time_steps * 0.50)
		val_size = int(time_steps * 0.25)
		test_size = int(time_steps * 0.25)

		train, val, test = self.__split_multidimentional_series(dataset, (0, train_size)),\
		self.__split_multidimentional_series(dataset, (train_size,train_size+val_size)),\
		self.__split_multidimentional_series(dataset, (train_size+val_size,time_steps))

		#create lookback
		X_train, y_train = self.__create_lookback_frame(train, look_back)
		X_val, y_val = self.__create_lookback_frame(val, look_back)
		X_test, y_test = self.__create_lookback_frame(test, look_back)

		print(X_train.shape)

		# X_train = np.reshape(X_train, (X_train.shape[2], X_train.shape[0], X_train.shape[1], X_train.shape[3]))
		# X_val = np.reshape(X_val, (X_val.shape[2], X_val.shape[0], X_val.shape[1], X_val.shape[3]))
		# X_test = np.reshape(X_test, (X_test.shape[2], X_test.shape[0], X_test.shape[1], X_test.shape[3]))

		return self.__create_spl_dframe(X_train, y_train, X_val, y_val, X_test, y_test), scalers