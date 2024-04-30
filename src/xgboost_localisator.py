'''
Script to train, test and save a xgboost tree model.
'''
# utils
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
import os

# graphical
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly

# Machine Learning
import xgboost as xgb

# FUNCTIONS

def hist_dist(y_true, ypred, save_path, type_set):
	"""
	Function to compute, show and save the figure of the distribution of the
	distance between the predicted and the true position of the earthquakes.

	Parameters
	----------
	y_true : numpy.ndarray
		True x, y and z positions of the earthquakes.
	ypred : numpy.ndarray
		Predicted x, y and z positions of the earthquakes.

	Returns
	-------
	distances_ : numpy.ndarray
		Distances per earthquakes between the prediction and the true one.

	"""
	distances_ = np.sum((ypred-y_true)**2, axis=1)**0.5
	counts_, bounds_ = np.histogram(distances_, 100)
	counts_pdf_ = counts_/np.sum(counts_)
	centers_b_ = (bounds_[:-1]+bounds_[1:])/2
	xlims = (bounds_[0], bounds_[-1])
	amplitude = (np.max(counts_pdf_)-np.min(counts_pdf_))*0.02
	upper = np.max(counts_pdf_)+amplitude

	median = np.median(distances_)
	average = np.mean(distances_)
	quantile95 = np.quantile(distances_, 0.95)

	plt.figure(figsize=(12, 5))
	plt.title('Euclidian distance: prediction - true / '+type_set)
	plt.vlines(centers_b_, 0, counts_pdf_, lw=3.5)
	plt.vlines(median, -1, 2, color='r',
				label='Median='+str(round(median, 2))+' m')

	plt.vlines(average, -1, 2, color='g',
				label='Mean='+str(round(average, 2))+' m')

	plt.vlines(quantile95, -1, 2, color='k',
				label='Quantile 95%='+str(round(quantile95, 2))+' m')

	plt.xlim(xlims[0], xlims[1])
	plt.ylim(0, upper)
	plt.xlabel('Distance pred-true (m)', fontsize='large')
	plt.ylabel('Distribution', fontsize='large')
	plt.legend(loc='upper right', fontsize=14, markerscale=2)
	plt.savefig(save_path+'distance_'+type_set+'.png', bbox_inches='tight')
	plt.show()

	return distances_


# CODES

# type of data used to train the model
use_type = 'raw'

# If we use the position of the stations as input features
use_position = True

# To save or not the train xgboost model in json file
save_model = True

# To compute or not the distribution of distance between the prediction and
# the three sets
make_distances = True

# To compute or not the 3d plot of predicted positions of the earthquakes
# from the asked set in `set_3d`
plot3d = True

# Set to use for the 3d plot
set_3d = 'test'

# Plot the features importance or not
show_fi = True

# VERIFICATIONS

if (use_type != 'raw')&(use_type != 'norm'):
	raise ValueError("The type of data used to train the model must be "+
					 "'norm' or 'raw'. Get `use_type = "+str(use_type)+"`")

if type(use_position) != bool:
	raise ValueError("`use_position` model must be a boolean. Get `type("+
					 "use_position) = "+str(type(use_position))+"`")

if type(save_model) != bool:
	raise ValueError("`save_model` model must be a boolean. Get `type("+
					 "save_model) = "+str(type(save_model))+"`")

if type(make_distances) != bool:
	raise ValueError("`make_distances` model must be a boolean. Get `type("+
					 "make_distances) = "+str(type(make_distances))+"`")

if type(plot3d) != bool:
	raise ValueError("`plot3d` model must be a boolean. Get `type(plot3d)"+
					 " = "+str(type(plot3d))+"`")

if (set_3d != 'train')&(set_3d != 'valid')&(set_3d != 'test'):
	raise ValueError("The 3d plots must be used on: 'train', 'valid' or "+
					 "'test'.Get `set_3d = "+str(set_3d)+"`")

if type(show_fi) != bool:
	raise ValueError("`show_fi` model must be a boolean. Get `type(show_fi"+
					 ") = "+str(type(show_fi))+"`")


# To only load needed data and save memory
if use_type == 'norm':
	train = np.load('../data/train_norm.npy', allow_pickle=True)[0]
	X_train, y_train = train['X_train'], train['y_train']

	valid = np.load('../data/valid_norm.npy', allow_pickle=True)[0]
	X_valid, y_valid = valid['X_valid'], valid['y_valid']

	test = np.load('../data/test_norm.npy', allow_pickle=True)[0]
	X_test, y_test = test['X_test'], test['y_test']
	if use_position == False:
		X_train = X_train[:, np.array([3, 7, 11, 15, 19])]
		X_valid = X_valid[:, np.array([3, 7, 11, 15, 19])]
		X_test = X_test[:, np.array([3, 7, 11, 15, 19])]

parameters = np.load('../data/parameters.npy', allow_pickle=True)[0]
min_X = parameters['min_X']
max_X = parameters['max_X']
sta_a = parameters['stations_cent']
n_fts = parameters['nfeatures']
ref_sta = parameters['ref_station']
limits = parameters['limites']
sta_a = parameters['stations_cent']
differ = parameters['differences']

train_raw = np.load('../data/train_raw.npy', allow_pickle=True)[0]
X_train_raw, y_train_raw = (train_raw['X_train_raw'],
							train_raw['y_train_raw'])

valid_raw = np.load('../data/valid_raw.npy', allow_pickle=True)[0]
X_valid_raw, y_valid_raw = (valid_raw['X_valid_raw'],
							valid_raw['y_valid_raw'])

test_raw = np.load('../data/test_raw.npy', allow_pickle=True)[0]
X_test_raw, y_test_raw = (test_raw['X_test_raw'], test_raw['y_test_raw'])

if use_position == False:
	X_train_raw = X_train_raw[:, np.array([3, 7, 11, 15, 19])]
	X_valid_raw = X_valid_raw[:, np.array([3, 7, 11, 15, 19])]
	X_test_raw = X_test_raw[:, np.array([3, 7, 11, 15, 19])]

# To automatically create a folder where the results will be saved, if asked
TODAY = datetime.today()
folder = '../xgboost/'
folder += str(TODAY.year)+'_'

if TODAY.month < 10:
	folder += '0'+str(TODAY.month)+'_'
else:
	folder += str(TODAY.month)+'_'

if TODAY.day < 10:
	folder += '0'+str(TODAY.day)+'_'
else:
	folder += str(TODAY.day)+'_'

if TODAY.hour < 10:
	folder += '0'+str(TODAY.hour)+'_'
else:
	folder += str(TODAY.hour)+'_'

if TODAY.minute < 10:
	folder += '0'+str(TODAY.minute)+'/'
else:
	folder += str(TODAY.minute)+'/'

if not os.path.exists(folder):
	os.makedirs(folder)

# Where results will be saved, if asked
print('Folder:', folder)

# Save in a text file the hyperparameters
with open(folder+'params.txt', 'w') as file:
	file.write('use_type: '+use_type+'\n')
	file.write('use_position: '+str(use_position)+'\n')
	file.write('save_model: '+str(save_model)+'\n')
	file.write('make_distances: '+str(make_distances)+'\n')
	file.write('plot3d: '+str(plot3d)+'\n')
	file.write('set_3d: '+set_3d+'\n')
	file.write('\n')

	file.write('Differences: '+str(differ)+'\n')
	file.write('\n')

	file.write('Train shape: '+str(X_train_raw.shape)+' ; '+str(y_train_raw.shape)+'\n')
	file.write('Valid shape: '+str(X_valid_raw.shape)+' ; '+str(y_valid_raw.shape)+'\n')
	file.write('Test shape: '+str(X_test_raw.shape)+' ; '+str(y_test_raw.shape)+'\n')
	file.write('\n')

# Creating the XGBoost model and training it
# Feel free to modify the parameter of the model
model = xgb.XGBRegressor(booster="dart")
if use_type == 'norm':
	model.fit(X_train, y_train)

	# Predictions
	prd_train = model.predict(X_train)
	prd_valid = model.predict(X_valid)
	prd_test = model.predict(X_test)

	# transforming the predictions to have meters
	prd_train[:, -1] = prd_train[:, -1] + 1
	prd_valid[:, -1] = prd_valid[:, -1] + 1
	prd_test[:, -1] = prd_test[:, -1] + 1

	prd_train = prd_train*(limits[:, 1]-limits[:, 0])
	prd_valid = prd_valid*(limits[:, 1]-limits[:, 0])
	prd_test = prd_test*(limits[:, 1]-limits[:, 0])

	prd_train = prd_train+limits[:, 0]
	prd_valid = prd_valid+limits[:, 0]
	prd_test = prd_test+limits[:, 0]

else:
	model.fit(X_train_raw, y_train_raw)

	# Predictions
	prd_train = model.predict(X_train_raw)
	prd_valid = model.predict(X_valid_raw)
	prd_test = model.predict(X_test_raw)

# Saving the xgboost model if asked
if save_model:
	model.save_model(folder+'xgb_model.json')

if make_distances:
	# Show and save the distribution of the differences between the predicted
	#  and the true position of the earthquakes
	distances_train = hist_dist(y_train_raw, prd_train, folder, 'train')
	distances_valid = hist_dist(y_valid_raw, prd_valid, folder, 'valid')
	distances_test = hist_dist(y_test_raw, prd_test, folder, 'test')

	# Save in a text file some results
	with open(folder+'params.txt', 'a') as file:
		file.write('train\n')
		file.write('Minimum: '+str(distances_train.min())+'\n')
		file.write('Median: '+str(np.median(distances_train))+'\n')
		file.write('Mean: '+str(np.mean(distances_train))+'\n')
		file.write('Quantile 0.95: '+str(np.quantile(distances_train,
					0.95))+'\n')
		file.write('Maximum: '+str(distances_train.max())+'\n')
		file.write('\n')

		file.write('valid\n')
		file.write('Minimum: '+str(distances_valid.min())+'\n')
		file.write('Median: '+str(np.median(distances_valid))+'\n')
		file.write('Mean: '+str(np.mean(distances_valid))+'\n')
		file.write('Quantile 0.95: '+str(np.quantile(distances_valid,
					0.95))+'\n')
		file.write('Maximum: '+str(distances_valid.max())+'\n')
		file.write('\n')

		file.write('test\n')
		file.write('Minimum: '+str(distances_test.min())+'\n')
		file.write('Median: '+str(np.median(distances_test))+'\n')
		file.write('Mean: '+str(np.mean(distances_test))+'\n')
		file.write('Quantile 0.95: '+str(np.quantile(distances_test,
					0.95))+'\n')
		file.write('Maximum: '+str(distances_test.max())+'\n')
		file.write('\n')

	# If the 3d plot was asked
	if plot3d:
		if set_3d == 'train':
			predictions = prd_train
			dists = distances_train
		if set_3d == 'valid':
			predictions = prd_valid
			dists = distances_valid
		if set_3d == 'test':
			predictions = prd_test
			dists = distances_test

		trace = go.Scatter3d(
					x=predictions[:, 0],
					y=predictions[:, 1],
					z=predictions[:, 2],
					mode='markers',
					marker={'size':2, 'color':dists,
							'colorscale':'Viridis','opacity': 0.6,
							'colorbar':{'thickness':20}},)

		layout = go.Layout(height=800, width=1000,
						   margin={'l': 0, 'r': 0, 'b': 0, 't': 0})

		data = [trace]
		plot_figure = go.Figure(data=data, layout=layout)
		plotly.offline.iplot(plot_figure)

		# To save the interactive figure in html file
		plot_figure.write_html(folder+set_3d+'_cube.html')

if use_position:
	fnames = np.array(['X_S1', 'Y_S1', 'Z_S1', 't_S1',
					   'X_S2', 'Y_S2', 'Z_S2', 't_S2',
					   'X_S3', 'Y_S3', 'Z_S3', 't_S3',
					   'X_S4', 'Y_S4', 'Z_S4', 't_S4',
					   'X_S5', 'Y_S5', 'Z_S5', 't_S5'])

else:
	fnames = np.array(['t_S1', 't_S2', 't_S3', 't_S4', 't_S5'])

importance = model.feature_importances_
rank = np.argsort(importance)
importance = importance[rank]
fnames = fnames[rank]
y_pos = np.arange(len(fnames))
with open(folder+'params.txt', 'a') as file:
	file.write('Feature importance:\n')
	for i in y_pos[::-1]:
		file.write(fnames[i]+': '+str(importance[i])+'\n')

	file.write('\n')

if show_fi:
	plt.figure(figsize=(8, 12))
	plt.grid(True, zorder=1)
	plt.hlines(y_pos, 0, importance, lw=10)
	plt.xlabel('Feature importance', fontsize=16)
	plt.xlim(0, np.max(importance)*1.02)
	plt.xticks(fontsize=15)
	plt.yticks(y_pos, fnames, fontsize=15)
	plt.savefig(folder+'feature_importance.png', bbox_inches='tight')
	plt.show()
