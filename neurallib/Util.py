import numpy as np
from sklearn.metrics import roc_curve

import sys, os, fnmatch, json
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.style

ax = 0

def extract_final_losses(history):
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	idx_min_val_loss = np.argmin(val_loss)
	return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history, arch_idx = None):
	print("Ploting training error curves...")

	train_loss = history.history['loss']
	val_loss = history.history['val_loss']

	mpl.style.use('default')
	fig, ax = plt.subplots()
	ax.grid(False)
	ax.plot(train_loss, label = 'Train')
	ax.plot(val_loss, label = 'Validation')
	ax.set(title = 'Training and Validation Error Curves', xlabel = 'Epochs', ylabel = 'Loss (MSE)')
	ax.legend()

	#plt.show()
	file = "plot.png"
	if arch_idx:
		file = "arch_" + str(arch_idx) + "_" + file

	file = "results/" + file

	fig.savefig(file)

def plot_roc_curve(y_test, y_pred, arch_idx = None):
	print("Ploting ROC curve...")
	fpr_net, tpr_net, _ = roc_curve(y_test, y_pred)

	mpl.style.use('default')
	fig, ax = plt.subplots()
	ax.grid(False)
	ax.plot([0, 1], [0, 1], 'k--')
	ax.plot(fpr_net, tpr_net, label='net1')
	ax.set(title = 'ROC Curve', xlabel = 'False positive rate', ylabel = 'True positive rate')

	#plt.show()
	file = "roc.png"
	if arch_idx:
		file = "arch_" + str(arch_idx) + "_" + file

	file = "results/" + file

	fig.savefig(file)

def plot_scatter_matrix(dset, hue):
	sns.set(color_codes=True)
	g = sns.pairplot(dset, hue=hue, vars=["f1", "f2", "f3", "f4", "f5", "f6"], markers=["o", "x"])
	g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
	g.savefig("./results/dset_scat_matrix.png")

def plot_boxPlot(dset):
	sns.set(color_codes=True)
	plt.figure()
	axes = dset.boxplot(by='target', figsize=(12, 6), return_type='axes')
	for ax in axes:
		ax.set_ylim(-1.4, 10)
	
	plt.savefig("./results/dset_boxplot.png")

def plot_series(dset_full, train_predict, val_predict, test_predict, look_back, scaler, dimension='x', arch_idx=None):
	print("Plotting serie curve...")

	mpl.style.use('default')
	fig, ax = plt.subplots()
	ax.grid(False)

	# shift train predictions for plotting
	train_predict_plot = np.empty_like(dset_full)
	train_predict_plot[:, :] = np.nan
	train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
	# shift validation predictions for plotting
	val_predict_plot = np.empty_like(dset_full)
	val_predict_plot[:, :] = np.nan
	val_predict_plot[len(train_predict)+(look_back):len(val_predict)+len(train_predict)+(look_back), :] = val_predict
	# shift test predictions for plotting
	test_predict_plot = np.empty_like(dset_full)
	test_predict_plot[:, :] = np.nan
	test_predict_plot[len(dset_full)-len(test_predict):len(dset_full), :] = test_predict
	# plot baseline and predictions
	plt.plot(dset_full)
	plt.plot(train_predict_plot, '--')
	plt.plot(val_predict_plot, '--')
	plt.plot(test_predict_plot, '--')

	#plt.show()
	file = dimension + "_serie.png"
	if arch_idx:
		file = "arch_" + str(arch_idx) + "_" + file

	file = "results/" + file

	fig.savefig(file)

def ouput_series_file(data, file_path):
	if not data.shape[0] == 2:
		data = np.array([data[:,0], data[:,1]])
	data = data.tolist()
	#data = [[data[0][i] for i in range(10)], [data[1][i] for i in range(10)]]
	file = open(file_path, 'w')
	file.write(str(data))
	file.close()

# code gently given by marcelsan #
def navigate_all_files(root_path, patterns):
    """
    A generator function that iterates all files that matches the given patterns
    from the root_path.
    """
    for root, dirs, files in os.walk(root_path):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                yield os.path.join(root, filename)

def get_all_files(root_path, patterns):
    """
    Returns a list of all files that matches the given patterns from the
    root_path.
    """
    ret = []
    for filepath in navigate_all_files(root_path, patterns):
        ret.append(filepath)
    return ret
##################################
def render_frame(i, data, points, message_text, train_size, validation_size, test_size):
	# if(i < train_size):
	# 	message_text.set_text('Training @sample = ' + str(i))
	# elif(i >= train_size and i < train_size+validation_size):
	# 	message_text.set_text('Validation @sample = ' + str(i))
	# elif(i >= train_size+validation_size):
	# 	message_text.set_text('Test @sample = ' + str(i))

	for j,p in enumerate(points):
		p.set_data(data[j][0][i], data[j][1][i])
	return points

def make_video_from_series( train_size, validation_size, test_size, output_path=""):
	# Simulation Path
	global ax
	print("Creating video...")
	path = output_path
	output_name = output_path + 'series_result.mp4'
	files = get_all_files(path, "*.xyz")

	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=200, bitrate=1600)

	mpl.style.use('default')
	fig, ax = plt.subplots()
	ax = plt.axes(xlim=(0, 105), ylim=(0, 68))
	ax.grid(False)

	files = [open(f) for f in files]
	data = np.array([json.loads(f.read()) for f in files])

	message_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
	
	points=[]
	for index in range(data.shape[1]):
		if(index % 2) == 0:
			pobj, = ax.plot(0, 0, 'bo')
		else:	
			pobj, = ax.plot(0, 0, 'rx')
		points.append(pobj)

	line_ani = animation.FuncAnimation(fig, render_frame, frames=data.shape[2],
											fargs=(data, points, message_text, train_size,
											validation_size, test_size), interval=5, blit=True,
											save_count=0, repeat=False)

	plt.show()
	#line_ani.save(output_name, writer=writer)
