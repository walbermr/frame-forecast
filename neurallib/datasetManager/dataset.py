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
			main = main[main.tagid == 9]	#get specific tag
			d_set_aux = {}
			for key in self.headers:
				d_set_aux[key] = list(map(lambda x: [x], main[key].values))

			self.dataframe, self.scaler = self.series_dataset(d_set_aux['x_pos'], look_back)

	def __create_spl_dframe(self, a, b, c, d, e, f):
		return {'X_train': a,
				'y_train': b,
				'X_test': c,
				'y_test': d,
				'X_val': e,
				'y_val': f}

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
		dataX, dataY = [], []
		for i in range(len(data)-look_back-1):
			dataX.append(data[i:(i+look_back), 0])
			dataY.append(data[i+look_back, 0])
			
		return np.array(dataX), np.array(dataY)

	def series_dataset(self, dataset, look_back):
		#normalize the dataset
		scaler = MinMaxScaler(feature_range=(0, 1))
		dataset = scaler.fit_transform(dataset)
		# split into train and test sets
		train_size = int(len(dataset) * 0.50)
		val_size = int(len(dataset) * 0.25)
		test_size = int(len(dataset) * 0.25)
		
		train, val, test = dataset[0:train_size,:],\
		dataset[train_size:train_size+val_size,:],\
		dataset[train_size+val_size:len(dataset),:]

		X_train, y_train = self.__create_lookback_frame(train, look_back)
		X_val, y_val = self.__create_lookback_frame(val, look_back)
		X_test, y_test = self.__create_lookback_frame(test, look_back)

		# reshape input to be [samples, time steps, features]
		X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
		X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
		X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

		return self.__create_spl_dframe(X_train, y_train, X_val, y_val, X_test, y_test), scaler