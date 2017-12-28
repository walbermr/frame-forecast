from sklearn.cluster import KMeans
from .datasetManager.dataframes import *
import numpy as np
from random import randint
from imblearn.over_sampling import SMOTE

def UniformSampling(df1, df2):

	ordered_dsets = get_ordered_dataframes(df1, df2)

	smallerdset = ordered_dsets['small']
	biggerdset = ordered_dsets['big']

	sizelarger, sizesmaller = get_dataframes_sizes(biggerdset, smallerdset)

	ratio = int(sizelarger / sizesmaller) + 1
	delta = sizelarger - (sizesmaller * ratio)

	for key in ['X_train', 'y_train']:
		smallerdset[key] = np.repeat(smallerdset[key], ratio, axis = 0)

		# remove rows if necessary
		if delta < 0:
			smallerdset[key] = smallerdset[key][0:delta]

	return smallerdset, biggerdset

def KMeansSampling(df1, df2):
	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']

	(sizelarger, sizesmaller) = get_dataframes_sizes(biggerdset, smallerdset)
	kmeans = KMeans(n_clusters = sizesmaller, random_state = 0).fit(biggerdset["X_train"])

	biggerdset["X_train"] = kmeans.cluster_centers_
	biggerdset["y_train"] = np.resize(biggerdset["y_train"], biggerdset["X_train"].shape[0])

	return smallerdset, biggerdset

def RandomSampling(df1, df2):
	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']

	(sizelarger, sizesmaller) = get_dataframes_sizes(biggerdset, smallerdset)

	for i in range(0, (sizelarger - sizesmaller)):
		index = randint(0, sizesmaller)
		for key in ['X_train', 'y_train']:
			smallerdset[key] = np.append(smallerdset[key], [smallerdset[key][index]], axis = 0)

	return smallerdset, biggerdset

def SMOTESampling(df1, df2):
	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']

	X_train = np.concatenate((smallerdset['X_train'], biggerdset['X_train']), axis = 0)
	y_train = np.concatenate((smallerdset['y_train'], biggerdset['y_train']), axis = 0)

	X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)

	biggerdset['X_train'] = X_resampled
	biggerdset['y_train'] = y_resampled

	# Override smaller dataset with a empty array
	smallerdset['X_train'] = np.empty([1, biggerdset['X_train'].shape[1]])[0:0, :]
	smallerdset['y_train'] = np.array([])

	return smallerdset, biggerdset
