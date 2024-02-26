# -*- coding: utf-8 -*-
"""
This module contain functions to plot the state of the search and the history
of the localisation of the earthquake.
"""
#import usefull library
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
def plot_dict_stations(stations, event=None, samples=None, save_path=None):
	"""
	Function to plot the position of the station, the true event and of the
	tested earthquake(s) model(s) if given.

	Parameters
	----------
	stations : dict
		Dictionary containing station data.
	event : dict
		Dictionary containing event data. The default is None.
	samples : dict, optional
		Dictionary containing samples data. The default is None.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	plt.figure(figsize=(12, 8))
	plt.title('Map of stations and test events', fontsize=20)
	plt.grid(True, zorder=1)
	c = 0
	for i in stations:
		if c == 0:
			plt.plot(stations[i]['X'], stations[i]['Y'], 'bv',
					 label='Station')

		else:
			plt.plot(stations[i]['X'], stations[i]['Y'], 'bv')

		c += 1
		plt.text(stations[i]['X'], stations[i]['Y']+10, i, fontsize=15,
				 ha='center')

	if type(event) == dict:
		plt.plot(event['X'], event['Y'], 'r*', label='Event')
		plt.text(event['X'], event['Y']+10, 'E', ha='center')

	c = 0
	if type(samples) == np.ndarray:
		for i in samples:
			if c == 0:
				plt.plot(samples[i]['X'], samples[i]['Y'], 'gv', zorder=2,
						 label='Best model(s)')

			else:
				plt.plot(samples[i]['X'], samples[i]['Y'], 'gv', zorder=2)

			c += 1

	plt.legend()
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_of_stations_dict.png',
					bbox_inches='tight')

	plt.show()

def plot_vect_stations(stations, event=None, samples=None, save_path=None):
	"""
	Function to plot the position of the station, the true event and of the
	tested earthquake(s) model(s) if given.

	Parameters
	----------
	stations : numpy.ndarray
		Numpy array containing station data.
	event : numpy.ndarray
		Numpy array containing event data. The default is None.
	samples : numpy.ndarray
		Numpy array containing samples data. The default is None.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	plt.figure(figsize=(12, 8))
	plt.title('Map of stations and test events', fontsize=20)
	plt.grid(True, zorder=1)
	plt.plot(stations[:, 0], stations[:, 1], 'bv', zorder=3, label='Station')
	for i in range(stations.shape[0]):
		plt.text(stations[i, 0], stations[i, 1]+10, 'S'+str(i),
				 fontsize=15, zorder=4, ha='center')

	if type(event) == np.ndarray:
		plt.plot(event[0], event[1], 'r*', label='Event')
		plt.text(event[0]+5, event[1]+10, 'E', ha='center')

	if type(samples) == np.ndarray:
		plt.plot(samples[:, 0], samples[:, 1], 'gv', label='Best sample(s)',
				 zorder=2)

	plt.legend()
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_of_stations_vect.png',
					bbox_inches='tight')

	plt.show()

def plot_history_dict(stations, test, history, loss, true_event=None,
					  samp_rate=60, save_path=None):
	"""
	Function to plot the evolution of the variables.

	Parameters
	----------
	stations : dict
		Dictionary containing station data.
	test : dict
		Best earthquake model data with the lower rmse.
	history : numpy.ndarray
		A 2d array listing the evolution of the variables of the model.
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and
		the data.
	true_event : dict, optional
		True event to locate. The default is None.
	samp_rate : int, optional
		An int to plot only a fraction of the position on the x and y map.
		The default is 60.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	x_axis = np.arange(loss.shape[0])[::samp_rate]
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	# RMSE from each epochs
	plt.figure()
	plt.title('Cost function (loss)', fontsize=12)
	plt.plot(x_axis, loss[::samp_rate], color='steelblue')
	plt.plot([x_axis[-1], len(loss)], [loss[x_axis[-1]], loss[-1]],
			 color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Root Mean Square Error', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'loss_hist_dict.png', bbox_inches='tight')

	plt.show()

	# Time before the earthquake was recorded by the reference station (in s)
	plt.figure()
	plt.title('Time variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 3], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 3], history[-1, 3]], color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Time (s)', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'time_hist_dict.png', bbox_inches='tight')

	plt.show()

	# Depth in relation to the reference station (in m)
	plt.figure()
	plt.title('Z variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 2], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 2], history[-1, 2]], color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Meters', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'depth_hist_dict.png', bbox_inches='tight')

	plt.show()

	plt.figure(figsize=(10, 10))
	plt.title('Map of stations, epicenter and path from gradient descent',
			  fontsize=12)

	plt.grid(True, zorder=1)
	c = 0
	for i in stations:
		if c == 0:
			plt.plot(stations[i]['X'], stations[i]['Y'], 'bv', ms=5,
					 label='Station')

		else:
			plt.plot(stations[i]['X'], stations[i]['Y'], 'bv', ms=5)

		c += 1
		plt.text(stations[i]['X'], stations[i]['Y']+20, i, fontsize=15,
				 ha='center')

	if type(true_event) == dict:
		plt.plot(true_event['X'], true_event['Y'], 'r*', label='Event',
				 zorder=2)

		plt.text(true_event['X'], true_event['Y']+20, 'E', zorder=3,
				 ha='center', va='center')

	plt.plot(history[::samp_rate, 0], history[::samp_rate, 1], 'g.', ms=0.5)
	plt.plot(test['X'], test['Y'], 'gv', zorder=2, label='Best model')
	plt.text(test['X'], test['Y']+20, 'U', zorder=3, ha='center')

	plt.legend()
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_hist_dict.png', bbox_inches='tight')

	plt.show()

def plot_history_vect(stations, test, history, loss, true_event=None,
					  samp_rate=60, save_path=None):
	"""
	Function to plot the evolution of the variables.

	Parameters
	----------
	stations : numpy.ndarray
		Numpy array containing station data.
	test : numpy.ndarray
		Best earthquake model data with the lower rmse.
	history : numpy.ndarray
		A 2d array listing the evolution of the variables of the model.
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and
		the data.
	true_event : numpy.ndarray, optional
		True event to locate. The default is None.
	samp_rate : int, optional
		An int to plot only a fraction of the position on the x and y map.
		The default is 60.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	x_axis = np.arange(loss.shape[0])[::samp_rate]
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	# RMSE from each epochs
	plt.figure()
	plt.title('Cost function (loss)', fontsize=12)
	plt.plot(x_axis, loss[::samp_rate], color='steelblue')
	plt.plot([x_axis[-1], len(loss)], [loss[x_axis[-1]], loss[-1]],
			 color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Root Mean Square Error', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'loss_hist_vect.png', bbox_inches='tight')

	plt.show()

	# Time before the earthquake was recorded by the reference station (in s).
	plt.figure()
	plt.title('Time variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 3], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 3], history[-1, 3]], color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Time (s)', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'time_hist_vect.png', bbox_inches='tight')

	plt.show()

	# Depth in relation to the reference station (in m)
	plt.figure()
	plt.title('Z variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 2], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 2], history[-1, 2]], color='steelblue')

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Meters', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'depth_hist_vect.png', bbox_inches='tight')

	plt.show()

	plt.figure(figsize=(10, 10))
	plt.title('Map of stations, epicenter and path from gradient descent',
			  fontsize=12)

	plt.grid(True, zorder=1)
	plt.plot(stations[:, 0], stations[:, 1], 'bv', ms=5, zorder=2, label='Station')

	for i in range(len(stations)):
		plt.text(stations[i, 0], stations[i, 1]+20, 'S'+str(i),
				 zorder=3, ha='center')

	if type(true_event) == np.ndarray:
		plt.plot(true_event[0], true_event[1], 'r*', zorder=2, label='Event')
		plt.text(true_event[0], true_event[1]+20, 'E', zorder=3,
				 ha='center')

	plt.plot(history[::samp_rate, 0], history[::samp_rate, 1], 'g.', ms=0.5)
	plt.plot(test[0], test[1], 'gv', zorder=2, label='Best model')
	plt.text(test[0], test[1]+20, 'U', zorder=3, ha='center')

	plt.legend()
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_hist_vect.png', bbox_inches='tight')

	plt.show()

def plot_history_vect_ed(stations, history, loss, true_event=None,
						 samp_rate=60, save_path=None):
	"""
	Function to plot the evolution of the variables for the ensemble descent
	method.

	Parameters
	----------
	stations : numpy.ndarray
		Numpy array containing station data.
	test : numpy.ndarray
		Best earthquake model data with the lower rmse.
	history : numpy.ndarray
		A 2d array listing the evolution of the variables of the model.
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and the
		data.
	true_event : numpy.ndarray, optional
		True event to locate. The default is None.
	samp_rate : int, optional
		An int to plot only a fraction of the position on the x and y map.
		The default is 60.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	x_axis = np.arange(loss.shape[0])[::samp_rate]
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	# RMSE from each epochs
	plt.figure()
	plt.title('Cost function (loss)', fontsize=12)
	plt.plot(loss, 'b', lw=1)
	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Root Mean Square Error', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'loss_hist_ed.png', bbox_inches='tight')

	plt.show()

	# Time before the earthquake was recorded by the reference station (in s)
	plt.figure()
	plt.title('Time variations', fontsize=12)
	plt.plot(history[:, :, 3], 'b', lw=1)
	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Time (s)', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'time_hist_ed.png', bbox_inches='tight')

	plt.show()

	# Depth in relation to the reference station (in m)
	plt.figure()
	plt.title('Z variations', fontsize=12)
	plt.plot(history[:, :, 2], 'b', lw=1)
	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Meters', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'depth_hist_ed.png', bbox_inches='tight')

	plt.show()

	# X-Y axis
	plt.figure(figsize=(10, 10))
	plt.title('Map of stations, epicenter and paths from gradient descent',
			  fontsize=12)

	plt.grid(True, zorder=1)
	plt.plot(stations[:, 0], stations[:, 1], 'bv', ms=5, zorder=2, label='Station')
	for i in range(len(stations)):
		plt.text(stations[i, 0], stations[i, 1]+20, 'S'+str(i), zorder=3,
				 ha='center')

	if type(true_event) == np.ndarray:
		plt.plot(true_event[0], true_event[1], 'r*', zorder=2, label='Event')
		plt.text(true_event[0], true_event[1]+20, 'E', zorder=3, ha='center')

	c = 0
	for i in range(history.shape[1]):
		plt.plot(history[::samp_rate, i, 0], history[::samp_rate, i, 1],
				 'g.', ms=0.5)

		if c == 0:
			plt.plot(history[np.argmin(loss[:, i]), i, 0],
					 history[np.argmin(loss[:, i]), i, 1], 'gv', zorder=2,
					 label='Best models')

		else:
			plt.plot(history[np.argmin(loss[:, i]), i, 0],
					 history[np.argmin(loss[:, i]), i, 1], 'gv', zorder=2)

		c += 1
		plt.text(history[np.argmin(loss[:, i]), i, 0],
				 history[np.argmin(loss[:, i]), i, 1]+20, 'U'+str(i),
				 ha='center', zorder=3)

	plt.legend()
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_hist_ed.png', bbox_inches='tight')

	plt.show()

def custom_2d_hist(array, bins=(60, 100)):
	"""
	A custom function to compute 2d hitogram on the evolution of a population
	of models.

	Parameters
	----------
	array : numpy.ndarray
		A 2d array. Shape aof the array should be : (number of epoch, number
		of model)
	bins : tuple, optional
		Number of cell in the two axis direction. The first axis is for the
		distribution of values took by the mode at the i-th epoch. The
		default is (60, 100).

	Returns
	-------
	pdf : numpy.ndarray
		A 2d array with the distribution of the values took by the models
		through their evolution.
	scale_x : numpy.ndarray
		A 1d array, with the boundary values of the bins through the time
		axis.
	scale_y : numpy.ndarray
		A 1d array, with the boundary values of the bins through the values
		axis.

	"""
	scale_x = np.linspace(0, array.shape[0], bins[1]+1)
	scale_y = np.linspace(np.min(array), np.max(array), bins[0]+1)
	pdf = np.zeros(bins)
	for i in range(bins[1]):
		pdf[:, i] = np.histogram(
						np.ravel(array[int(scale_x[i]):int(scale_x[i+1])]),
								bins=bins[0],
								range=(scale_y[0], scale_y[-1]))[0]/(
										int(scale_x[i+1])-int(scale_x[i]))

	pdf[pdf == 0] = np.nan
	return pdf, scale_x, scale_y

def show_density_history_ed(stations, history, loss, true_event=None,
							bins_evol=(60, 100), bins_map=(100, 100),
							save_path=None):
	"""
	Function to show the evolution of the variables for the ensemble descent
	method through density plots.

	Parameters
	----------
	stations : numpy.ndarray
		Numpy array containing station data.
	history : numpy.ndarray
		A 2d array listing the evolution of the variables of the model.
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and the
		data.
	true_event : numpy.ndarray, optional
		True event to locate. The default is None.
	bins_evol : tuple, optional
		Number of cell in the two axis direction. The first axis is for the
		distribution of values took by the mode at the i-th epoch. The
		default is (60, 100).
	bins_map : tuple, optional
		Number of cell in the x and y axis direction for the distribution map
		of the position of the trained models.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	# Loss errors
	pdf_loss, sc_epoch, sc_loss = custom_2d_hist(loss, bins_evol)
	plt.figure(figsize=(12, 12))
	plt.title('Cost function (loss)', fontsize=12)
	plt.grid(True)
	plt.imshow(pdf_loss[::-1], interpolation='none')
	plt.xticks(np.linspace(0, len(sc_epoch), 11),
			   np.round(np.linspace(sc_epoch[0], sc_epoch[-1], 11), 0))

	plt.yticks(np.linspace(len(sc_loss), 0, 11),
			   np.round(np.linspace(sc_loss[0], sc_loss[-1], 11), 6))

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Root Mean Square Error', fontsize=11)
	plt.colorbar(shrink=0.4, label='Number of model per cell')
	if type(save_path) == str:
		plt.savefig(save_path+'loss_hist_pdf_ed.png', bbox_inches='tight')

	plt.show()

	# Time before the earthquake was recorded by the reference station (in s)
	pdf_time, sc_epoch, sc_time = custom_2d_hist(history[:, :, 3], bins_evol)
	plt.figure(figsize=(12, 12))
	plt.title('Time variations', fontsize=12)
	plt.grid(True)
	plt.imshow(pdf_time[::-1], interpolation='none')
	plt.xticks(np.linspace(0, len(sc_epoch), 11),
			   np.round(np.linspace(sc_epoch[0], sc_epoch[-1], 11), 0))

	plt.yticks(np.linspace(len(sc_loss), 0, 11),
			   np.round(np.linspace(sc_time[0], sc_time[-1], 11), 6))

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Time (s)', fontsize=11)
	plt.colorbar(shrink=0.4, label='Number of model per cell')
	if type(save_path) == str:
		plt.savefig(save_path+'time_hist_pdf_ed.png', bbox_inches='tight')

	plt.show()

	# Depth in relation to the reference station (in m)
	pdf_depth, sc_epoch, sc_depth = custom_2d_hist(history[:, :, 2], bins_evol)
	plt.figure(figsize=(12, 12))
	plt.title('Z variations', fontsize=12)
	plt.grid(True)
	plt.imshow(pdf_depth[::-1], interpolation='none')
	plt.xticks(np.linspace(0, len(sc_epoch), 11),
			   np.round(np.linspace(sc_epoch[0], sc_epoch[-1], 11), 0))

	plt.yticks(np.linspace(len(sc_depth), 0, 11),
			   np.round(np.linspace(sc_depth[0], sc_depth[-1], 11), 6))

	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Meters', fontsize=11)
	plt.colorbar(shrink=0.4, label='Number of model per cell')
	if type(save_path) == str:
		plt.savefig(save_path+'depth_hist_pdf_ed.png', bbox_inches='tight')

	plt.show()

	# X-Y axis
	pdf_map, sc_x_axis, sc_y_axis = np.histogram2d(history[-1, :, 1], 
												   history[-1, :, 0],
												   bins=bins_map,
												   range=((
													  np.min(stations[:, 0]),
													  np.max(stations[:, 0])),
													 (np.min(stations[:, 1]),
													  np.max(stations[:, 1]))
												   ))

	pdf_map = pdf_map[::-1]
	pdf_map[pdf_map == 0] = np.nan
	de = np.max(stations, axis=1)-np.min(stations, axis=1)
	xlims = (np.min(stations[:, 0])-de[0]*0.05,
			 np.max(stations[:, 0])+de[0]*0.05,)

	ylims = (np.min(stations[:, 1])-de[1]*0.07,
			 np.max(stations[:, 1])+de[1]*0.07,)

	plt.figure(figsize=(10, 10))
	plt.title('Map of stations and epicenters', fontsize=12)

	plt.imshow(pdf_map, interpolation='none', extent=[np.min(stations[:, 0]),
													  np.max(stations[:, 0]),
													  np.min(stations[:, 1]),
													  np.max(stations[:, 1])])

	plt.grid(True, zorder=1)
	plt.plot(stations[:, 0], stations[:, 1], 'bv', ms=5, zorder=2, label='Station')
	for i in range(len(stations)):
		plt.text(stations[i, 0], stations[i, 1]+15, 'S'+str(i), zorder=3,
				 ha='center')

	if type(true_event) == np.ndarray:
		plt.plot(true_event[0], true_event[1], 'r*', zorder=2, label='Event')
		plt.text(true_event[0], true_event[1]+15, 'E', zorder=3, ha='center')

	plt.plot(history[-1, :, 0], history[-1, :, 1], 'g.', label='Trained models')
	plt.colorbar(shrink=0.7, label='Number of model per cell')
	plt.legend()
	plt.xlim(xlims[0], xlims[1])
	plt.ylim(ylims[0], ylims[1])
	plt.xlabel('X (in meters)', fontsize=15)
	plt.ylabel('Y (in meters)', fontsize=15)
	if type(save_path) == str:
		plt.savefig(save_path+'map_hist_dense_ed.png', bbox_inches='tight')

	plt.show()

def plot_diff_cost_evol(loss, save_path=None):
	"""
	Function to plot the change in variation of the rmse through the epochs.

	Parameters
	----------
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and the
		data.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	plt.figure(figsize=(14, 5))
	plt.title('Change in variation of rmse', fontsize=12)
	plt.grid(True, zorder=2)
	plt.plot(np.diff(loss), zorder=3)
	plt.xlabel('Iterations', fontsize=11)
	plt.ylabel('Delta Root Mean Square Error', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'diff_cost_evol.png', bbox_inches='tight')

	plt.show()

def plot_diff_cost_evol_ed(loss, save_path=None):
	"""
	Function to plot the change in variation of the rmse through the epochs for
	the ensemble descent method.

	Parameters
	----------
	loss : numpy.ndarray
		A 1d array listing the evolution of the rmse between the model and the
		data.
	save_path : str, optional
		Path to the folder wher the graph will be saved. The default is None.

	Returns
	-------
	None.

	"""
	if type(save_path) == str:
		if save_path[-1] != '/':
			save_path = save_path+'/'

	for i in range(loss.shape[1]):
		plt.figure(figsize=(14, 5))
		plt.title('Change in variation of rmse', fontsize=12)
		plt.grid(True, zorder=2)
		plt.plot(np.diff(loss), zorder=3)
		plt.xlabel('Iterations', fontsize=11)
		plt.ylabel('Delta Root Mean Square Error', fontsize=11)
		if type(save_path) == str:
			plt.savefig(save_path+'diff_cost_evol_'+str(i)+'.png', bbox_inches='tight')

		plt.show()
