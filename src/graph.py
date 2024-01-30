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
	Function to plot the position of the station and of the tested
	earthquake(s) if given.

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
	for i in stations:
		plt.plot(stations[i]['X'], stations[i]['Y'], 'o')
		plt.text(stations[i]['X']+30, stations[i]['Y']+20, i, fontsize=15)

	if type(event) == dict:
		plt.plot(event['X'], event['Y'], 'r*')
		plt.text(event['X']+30, event['Y']+30, 'E', ha='center', va='center')

	if type(samples) == np.ndarray:
		for i in samples:
			plt.plot(samples[i]['X'], samples[i]['Y'], 'gv', zorder=2)

	plt.xlabel('X (in metres)', fontsize=15)
	plt.ylabel('Y (in metres)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_of_stations.png', bbox_inches='tight')

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
		A 1d array listing the evolution of the rmse between the model and the
		data.
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

	plt.xlabel('iterations', fontsize=11)
	plt.ylabel('rmse', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'loss_hist.png', bbox_inches='tight')

	plt.show()

	# Time before the earthquake was recorded by the reference station (in s).
	plt.figure()
	plt.title('time variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 3], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 3], history[-1, 3]], color='steelblue')

	plt.xlabel('iterations', fontsize=11)
	plt.ylabel('time (s)', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'time_hist.png', bbox_inches='tight')

	plt.show()

	# Depth in relation to the reference station (in m)
	plt.figure()
	plt.title('z variations', fontsize=12)
	plt.plot(x_axis, history[::samp_rate, 2], color='steelblue')
	plt.plot([x_axis[-1], len(loss)],
			 [history[x_axis[-1], 2], history[-1, 2]], color='steelblue')

	plt.xlabel('iterations', fontsize=11)
	plt.ylabel('meters', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'depth_hist.png', bbox_inches='tight')

	plt.show()

	plt.figure(figsize=(10, 10))
	plt.title('Map of stations, epicenter and path from gradient descent',
			  fontsize=12)

	plt.grid(True, zorder=1)
	for i in stations:
		plt.plot(stations[i]['X'], stations[i]['Y'], 'bo', zorder=2)
		plt.text(stations[i]['X']+0.4, stations[i]['Y'], i, zorder=3)

	if type(true_event) == dict:
		plt.plot(true_event['X'], true_event['Y'], 'r*', zorder=2)
		plt.text(true_event['X']+0.4, true_event['Y'], 'E', zorder=3)

	plt.plot(history[::samp_rate, 0], history[::samp_rate, 1], 'g.', ms=0.5)
	plt.plot(test['X'], test['Y'], 'gv', zorder=2)
	plt.text(test['X']+0.4, test['Y'], 'U', zorder=3)
	plt.xlabel('X (in metres)', fontsize=15)
	plt.ylabel('Y (in metres)', fontsize=15)
	plt.axis('equal')
	if type(save_path) == str:
		plt.savefig(save_path+'map_hist.png', bbox_inches='tight')

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
	plt.xlabel('iterations', fontsize=11)
	plt.ylabel('delta rmse', fontsize=11)
	if type(save_path) == str:
		plt.savefig(save_path+'diff_cost_evol.png', bbox_inches='tight')

	plt.show()
