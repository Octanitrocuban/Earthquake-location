# -*- coding: utf-8 -*-
"""
This module contains function to compute gradient descent through exploration
algorithms.
"""
import math as m
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
#=============================================================================
def print_sample_array(sample, before='', after=''):
	"""
	Function to plot samples array as dictionary.

	Parameters
	----------
	sample : numpy.ndarray
		Sample that we want to print.
	before : str, optional
		Text to be put before the sample value's. The default is ''.
	after : str, optional
		Text to be put after the sample value's. The default is ''.

	Returns
	-------
	None

	"""
	dico = {'X':sample[0], 'Y':sample[1], 'Z':sample[2], 't':sample[3]}
	print(before+'{}'.format(dico)+after)

def centrering(stations, key, event=None):
	"""
	Function to normalise (centralise) stations and event if given.

	Parameters
	----------
	stations : dict or numpy.ndarray
		Station locations and picked arrival time.
	key : str or int
		Item key to select the reference station.
	event : dict or numpy.ndarray
		Event to be normalised with the key reference station. The default
		is None.

	Returns
	-------
	stations : dict or numpy.ndarray
		Normalised (centralised) station relative to the reference station.
	event : dict or numpy.ndarray, optional
		Normalised (centralised) event relative to the reference station.

	"""
	if type(event) == dict:
		site_ref = deepcopy(stations[key])
		event['X'] = event['X']-site_ref['X']
		event['Y'] = event['Y']-site_ref['Y']
		event['Z'] = event['Z']-site_ref['Z']
		event['t'] = event['t']-site_ref['t']

	elif type(event) == np.ndarray:
		event = event-stations[key]

	if type(stations) == dict:
		site_ref = deepcopy(stations[key])
		for i in stations:
			stations[i]['X'] = stations[i]['X']-site_ref['X']
			stations[i]['Y'] = stations[i]['Y']-site_ref['Y']
			stations[i]['t'] = stations[i]['t']-site_ref['t']

		if type(event) == dict:
			return stations, event
		else:
			return stations

	elif type(stations) == np.ndarray:
		stations = stations-stations[key]
		if type(event) == np.ndarray:
			return stations, event
		else:
			return stations

	else:
		raise TypeError('stations must be dict or numpy.ndarray.')

def calc_misfit(stations, event, vp=4000):
	"""
	Calculate misfit between observed and calculated arrival times.

	Parameters
	----------
	stations : dict
		Station locations and picked arrival time.
	event : dict
		Event parameters : location and time {X, T, Z, t}.
	vp : float, optional
		Assumed constant P-wave velocity. The default is 4000.

	Returns
	-------
	misit : float
		The value of the root mean square error between the data recorded by
		the stations and the earthquake parameter proposed.

	"""
	misfit_sum_sq = 0.
	for key, station in stations.items():
		dist = m.sqrt((station['X'] - event['X'])**2
					 +(station['Y'] - event['Y'])**2
					 +(station['Z'] - event['Z'])**2)

		tp_est = event['t'] + dist / vp
		tp_obs = station['t']
		misfit_sum_sq += (tp_est - tp_obs)**2

	misfit = m.sqrt(misfit_sum_sq) / len(stations)
	return misfit

def gradient_descent(stations, event_test, n_iteration, ran_x=1, ran_y=1,
					  ran_z=1, ran_t=0.0005, l_r_m=0.2, patience=1000,
					  vp=4000):
	"""
	Function to make a gradient descent from a given test event and data from
	stations.

	Parameters
	----------
	stations : dict
		Station locations and picked arrival time.
	event_test : dict
		Event parameters : location and time {X, T, Z, t}.
	n_iteration : int
		Maximal number of iteration.
	ran_x : float, optional
		Specified learning rate for the X parameter. The default is 1.
	ran_t : float, optional
		Specified learning rate for the Y parameter. The default is 1.
	ran_z : float, optional
		Specified learning rate for the Z parameter. The default is 1.
	ran_t : float, optional
		Specified learning rate for the t parameter. The default is 0.0005.
	l_r_m : float, optional
		General learning rate. The default is 0.2.
	vp : float, optional
		Assumed constant P-wave velocity. The default is 4000.

	Returns
	-------
	event_test : dict
		Best event founded with the lower root mean square error.
	cost_story : numpy.ndarray
		Loss history at each iteration.
	history : numpy.ndarray
		Listing of the best event at each iteration.

	"""
	# Specified learning rate
	ran_x_lr = ran_x*l_r_m
	ran_y_lr = ran_y*l_r_m
	ran_z_lr = ran_z*l_r_m
	ran_t_lr = ran_t*l_r_m
	p3p = int(patience*3)
	p2p = int(patience*2)
	# Gives us as an indication the misfit before applying gradient descent
	misfit_ini = calc_misfit(stations, event_test, vp)
	print('Starting event = {}, starting misfit = {:.10f}'.format(
													event_test, misfit_ini))

	history = [[event_test['X'], event_test['Y'],
				event_test['Z'], event_test['t']]]

	cost_story = [calc_misfit(stations, event_test, vp)]
	for i in range(n_iteration):
		test = event_test.copy()
		test['X'] += ran_x_lr
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['X'] += ran_x_lr

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['X'] -= ran_x_lr

		test = event_test.copy()
		test['Y'] += ran_y_lr
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['Y'] += ran_y_lr

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['Y'] -= ran_y_lr

		test = event_test.copy()
		test['Z'] += ran_z_lr
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['Z'] += ran_z_lr

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['Z'] -= ran_z_lr

		test = event_test.copy()
		test['t'] += ran_t_lr
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['t'] += ran_t_lr

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['t'] -= ran_t_lr

		cost_story.append(calc_misfit(stations, event_test, vp))
		history.append([event_test['X'], event_test['Y'], event_test['Z'],
					 event_test['t']])

		if i > p3p:
			if (i%patience) == 0:
				# In cases where the convergence threshold is reached before
				# the n-th iterations are made, if the slope is sufficiently
				# low then we break the loop.
				if (np.min(cost_story[-p2p:-patience]) <= 
					np.min(cost_story[-patience:])):

					break

	cost_story = np.array(cost_story)
	history = np.array(history, dtype=object)
	misfit_fin = calc_misfit(stations, event_test, vp)
	print("Best event = {}, best misfit = {:.10f}".format(event_test,
															misfit_fin))

	return event_test, history, cost_story

def ensemble_descent(stations, n_samples, learning_rates, n_it, v_propag,
					 limits, patience=1000):
	"""
	Function to make a gradient descent with many samples in a vectorised
	method.

	Parameters
	----------
	stations : numpy.ndarray
		Station locations and picked arrival time.
	n_samples : int
		Number of sample with wich the gradient descent will be done.
	learning_rates : numpy.ndarray
		Learning rates of the parameters. If a rate is at 0 the parameter
		will not be updated.
	n_it : int
		Maximal number of iteration.
	v_propag : float
		Assumed constant P-wave velocity.
	limits : list
		List of list of float. Size is (n, 2). 'n' is the number of features.
		The first n-th value is the lower bound and the second is the upper
		bound of the random search.
	patience : int, optional
		Patience for the early stoping. The default is 1000.

	Returns
	-------
	loss : numpy.ndarray
		Loss history at each iteration.
	history : numpy.ndarray
		Listing of the best event at each iteration.

	"""
	p2p = int(patience*2)
	nxis = np.newaxis
	samps_pk = np.arange(n_samples)*16
	sample = np.array([
		np.random.uniform(limits[0][0], limits[0][1], n_samples),
		np.random.uniform(limits[1][0], limits[1][1], n_samples),
		np.random.uniform(limits[2][0], limits[2][1], n_samples),
		np.zeros(n_samples)-1]).T

	kernel = np.array([[ 1,  1,  1,  1], [-1,  1,  1,  1], [ 1, -1,  1,  1],
					   [ 1,  1, -1,  1], [ 1,  1,  1, -1], [-1, -1,  1,  1],
					   [-1,  1, -1,  1], [-1,  1,  1, -1], [ 1, -1, -1,  1],
					   [ 1, -1,  1, -1], [ 1,  1, -1, -1], [-1, -1, -1,  1],
					   [-1, -1,  1, -1], [-1,  1, -1, -1], [ 1, -1, -1, -1],
					   [-1, -1, -1, -1]])*learning_rates

	dist_v = cdist(stations[:, :3], sample[:, :3]).T
	pred_t = sample[:, 3:4] + dist_v / v_propag
	rmse = np.sum((pred_t-stations[:, 3])**2, axis=1)**.5 /len(stations)
	print('Starting misfit = '+str(np.mean(rmse)))
	loss = []
	history = []
	for i in range(n_it):
		# reaserch shape = (n samples*16, 4)
		reaserch = np.reshape(sample[:, nxis]+kernel, (n_samples*16, 4))
		# dist_v shape = (n samples, 16, n stations)
		dist_v = cdist(stations[:, :3], reaserch[:, :3]).T
		# pred_t shape = (n samples*16, n stations)
		pred_t = reaserch[:, 3:4] + dist_v / v_propag
		# rmse shape = (n samples*16)
		rmse = np.sum((pred_t-stations[:, 3])**2, axis=1)**.5 /len(stations)
		best = np.argmin(np.reshape(rmse, (n_samples, 16)), axis=1)+samps_pk
		loss.append(rmse[best])
		sample = reaserch[best]
		history.append(sample)
		if (i > p2p)&((i%patience) == 0):
			sub_loss = np.array(loss[-p2p:])
			pre_1 = np.mean(sub_loss[:patience], axis=0)
			pre_2 = np.mean(sub_loss[patience:], axis=0)
			if np.sum(pre_1 <= pre_2) == n_samples:
				break

	loss = np.array(loss)
	print("Best misfit = "+str(np.min(loss[-1])))
	best = np.argwhere(loss == np.min(loss))[-1]
	history = np.array(history)
	print_sample_array(history[best[0], best[1]], before='Best event = ')
	return history, loss

def monte_carlo(n_samples, limites, stations, vp=4000, sampling='random'):
	"""
	Function to compute a Monte Carlo method.

	Parameters
	----------
	n_samples : int
		Number of samples.
	limites : numpy.ndarray
		A 2d array listing the limites of the search. The shape is (n, 2).
		n is the number of features. The n-th line is: [lower, upper] bound.
	stations : numpy.ndarray
		Station locations and picked arrival time.
	vp : float, optional
		Assumed constant P-wave velocity. The default is 4000.
	sampling : str, optional
		Method to sampling the parameter space. If 'random', it will simply
		draw n_samples random values for each parameters. If 'grid_rand', it
		will first define a 4d grid with a total of cells <= n_samples. Then
		it will move randomly these samples within their cell. It can be
		['random', 'grid_rand']. The default is'random'.

	Returns
	-------
	samples : numpy.ndarray
		Listing of the best sample at each iteration.
	loss : numpy.ndarray
		Root mean square error of the sample tested.

	"""
	if sampling == 'random':
		samples = np.random.uniform(0, 1, (n_samples, 4))
		samples = samples*(limites[:, 1]-limites[:, 0])+limites[:, 0]

	elif sampling == 'grid_rand':
		sub_n = int(n_samples**(1/4))
		differ = ((limites[:, 1]-limites[:, 0])/(n_samples-1))/2
		samples = np.meshgrid(np.linspace(limites[0, 0], limites[0, 1], sub_n),
							  np.linspace(limites[1, 0], limites[1, 1], sub_n),
							  np.linspace(limites[2, 0], limites[2, 1], sub_n),
							  np.linspace(limites[3, 0], limites[3, 1], sub_n))

		samples = np.array([np.ravel(samples[0]), np.ravel(samples[1]),
							np.ravel(samples[2]), np.ravel(samples[3])]).T

		noise = np.random.uniform(-1, 1, samples.shape)*differ[np.newaxis]
		samples = samples+noise

	dist = cdist(stations[:, :3], samples[:, :3]).T
	tp_est = samples[:, 3:4] + dist / vp
	loss = np.sum((tp_est - stations[:, 3])**2, axis=1)**.5 /len(stations)
	print("Best misfit = "+str(np.min(loss)))
	best = samples[np.argmin(loss)]
	print_sample_array(best, before='Best event = ')
	return samples, loss

def deepening_grid_search(stations, sampling_f, limites, depth,
						  width_factor=2, vp=4000):
	"""
	Function to compute a deepening grid search method.

	Parameters
	----------
	stations : numpy.ndarray
		Station locations and picked arrival time.
	sampling_f : int
		Sampling frequence of the space of parameters to create samples. The
		number of samples will be sampling_f to the power four.
	limites : numpy.ndarray
		A 2d array listing the limites of the search. The shape is (n, 2).
		n is the number of features. The n-th line is: [lower, upper] bound.
	depth : int
		Depth of the search, i.e. the number of epoch.
	width_factor : float
		This is not learning rate ! This is a weight to counter the fast shrinked
		of the width of the grid used in searching. The default is 2.
	vp : float, optional
		Assumed constant P-wave velocity. The default is 4000.

	Returns
	-------
	samp_hist : numpy.ndarray
		Sample created and tested.
	cost_hist : numpy.ndarray
		Root mean square error history of the best sample at each iteration.

	"""
	cost_hist = np.zeros(depth)
	samp_hist = np.zeros((depth, 4))
	lims = np.copy(limites)
	for i in range(depth):
		delta = np.array([-(lims[:, 1]-lims[:, 0])/sampling_f,
						   (lims[:, 1]-lims[:, 0])/sampling_f]).T*width_factor

		samples = np.meshgrid(np.linspace(lims[0, 0], lims[0, 1], sampling_f),
							  np.linspace(lims[1, 0], lims[1, 1], sampling_f),
							  np.linspace(lims[2, 0], lims[2, 1], sampling_f),
							  np.linspace(lims[3, 0], lims[3, 1], sampling_f))

		samples = np.array([np.ravel(samples[0]), np.ravel(samples[1]),
							np.ravel(samples[2]), np.ravel(samples[3])]).T

		distances = cdist(stations[:, :3], samples[:, :3]).T
		t_estim = samples[:, 3:4] + distances / vp
		loss = np.sum((t_estim - stations[:, 3])**2, axis=1)**.5 /len(stations)
		current = samples[np.argmin(loss)]
		lims = current[:, np.newaxis]+delta
		samp_hist[i] = current
		cost_hist[i] = np.min(loss)

	print("Best misfit = "+str(np.min(loss)))
	best = samples[np.argmin(loss)]
	print_sample_array(best, before='Best event = ')
	return samp_hist, cost_hist
