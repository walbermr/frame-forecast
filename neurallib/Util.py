import numpy as np
from sklearn.metrics import roc_curve

import sys, os
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
import matplotlib
warnings.filterwarnings("ignore")

import seaborn as sns
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl


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
	plt.plot(train_predict_plot)
	plt.plot(val_predict_plot)
	plt.plot(test_predict_plot)

	#plt.show()
	file = dimension + "_serie.png"
	if arch_idx:
		file = "arch_" + str(arch_idx) + "_" + file

	file = "results/" + file

	fig.savefig(file)

def ouput_series_file(data, file_path):
	file = open(file_path, 'w')
	file.write(str(data))
	file.close()

# code gently given by marcelsan
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

def render_frame(num, data, line):
    xyz_file = open(files[num])
    
    _x = []
    _y = []

    for xyz in xyz_file:   
        _x.append(float(xyz.split()[0]))
        _y.append(float(xyz.split()[1]))

    line.set_data(_x, _y)
    xyz_file.close()

    return line,

def make_video_from_series(output_path=""):
	# Simulation Path
	print("Creating video...")
	path = output_path
	output_name = output_path + 'series_result.mp4'
	files = get_all_files(path, "*.xyz")
	size_files = len(files)

	# Set up formatting for the movie files
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=20, bitrate=1800)

	fig1 = plt.figure()

	data = []
	l, = plt.plot([], [], 'ro')

	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel('x')
	plt.title('Simulation Result')

	line_ani = animation.FuncAnimation(fig1, render_frame, size_files, fargs=(data, l),
									interval=50, blit=True)

	line_ani.save(output_name, writer=writer)
