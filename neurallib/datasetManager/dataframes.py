import numpy as np
import pandas as pd

def select_target(df, feat, value):
	return df[df[feat] == value]

def get_ordered_dataframes(df1, df2):
	sizes = get_dataframes_sizes(df1, df2)
	df1_size = sizes[0]
	df2_size = sizes[1]

	big, small = (df2, df1) \
					if(df2_size > df1_size) else \
				 (df1, df2)

	return {'big': big.copy(), 'small': small.copy()}

def get_dataframes_sizes(df1, df2):
	size1 = df1['X_train'].shape[0]
	size2 = df2['X_train'].shape[0]

	return	(size1, size2)