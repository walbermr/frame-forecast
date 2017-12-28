import numpy as np
import pandas as pd
from neurallib.datasetManager.dataframes import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from neurallib.Util import plot_scatter_matrix, plot_boxPlot

class DataSet:
	def __init__(self, path, headers, sampling):
		self.path = path
		self.headers = headers

		main = pd.read_csv(self.path, names = headers)
		main.drop_duplicates(inplace = True)

		if(PLOT):
			plot_scatter_matrix(main, 'target')
			plot_boxPlot(main)

		df1 = select_target(main, 'target', 0)
		df2 = select_target(main, 'target', 1)

		# Split dataset.
		df1 = self.split_dataframe(df1)
		df2 = self.split_dataframe(df2)

		df1, df2 = sampling(df1, df2)

		self.dataframe = self.concatenate_and_shuffle_dataset(df1, df2)

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