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
def centrering(stations, key, event=None):
	"""
	Function to normalise (centralise) stations and event if given.

	Parameters
	----------
	stations : dict
		Station locations and picked arrival time.
	key : str
		Item key to select the reference station.
	event : dict
		Event to be normalised with the key reference station. The default
		is None.

	Returns
	-------
	stations : dict or numpy.ndarray
		Normalised (centralised) station relative to the reference station.
	event : dict or numpy.ndarray, optional
		Normalised (centralised) event relative to the reference station.

	"""
	site_ref = deepcopy(stations[key])
	event['X'] = event['X']-site_ref['X']
	event['Y'] = event['Y']-site_ref['Y']
	event['Z'] = event['Z']-site_ref['Z']
	event['t'] = event['t']-site_ref['t']
	site_ref = deepcopy(stations[key])
	for i in stations:
		stations[i]['X'] = stations[i]['X']-site_ref['X']
		stations[i]['Y'] = stations[i]['Y']-site_ref['Y']
		stations[i]['t'] = stations[i]['t']-site_ref['t']
	if type(event) == dict:
		return stations, event
	else:
		return stations

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

def descente_gradient(stations, event_test, n_iteration, ranX=1, ranY=1,
					  ranZ=1, rant=0.0005, l_r_m=0.2, patience=1000,
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
	ranX : float, optional
		Specified learning rate for the X parameter. The default is 1.
	ranY : float, optional
		Specified learning rate for the Y parameter. The default is 1.
	ranZ : float, optional
		Specified learning rate for the Z parameter. The default is 1.
	rant : float, optional
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
	rant = rant*l_r_m
	ranX = ranX*l_r_m
	ranY = ranY*l_r_m
	ranZ = ranZ*l_r_m
	p3p = int(patience*3)
	p2p = int(patience*2)
	# Gives us as an indication the misfit before applying gradient descent
	misfit_ini = calc_misfit(stations, event_test, vp)
	# print('Starting event = {}, starting misfit = {:.10f}'.format(
	# 												event_test, misfit_ini))

	history = [[event_test['X'], event_test['Y'],
				event_test['Z'], event_test['t']]]

	cost_story = [calc_misfit(stations, event_test, vp)]
	for i in range(n_iteration):
		test = event_test.copy()
		test['X'] += ranX
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['X'] += ranX

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['X'] -= ranX

		test = event_test.copy()
		test['Y'] += ranY
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['Y'] += ranY

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['Y'] -= ranY

		test = event_test.copy()
		test['Z'] += ranZ
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['Z'] += ranZ

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['Z'] -= ranZ

		test = event_test.copy()
		test['t'] += rant
		if (calc_misfit(stations, test, vp)-
			calc_misfit(stations, event_test, vp)) < 0:
			event_test['t'] += rant

		elif (calc_misfit(stations, test, vp)-
				calc_misfit(stations, event_test, vp)) > 0:
			event_test['t'] -= rant

		cost_story.append(calc_misfit(stations, event_test, vp))
		history.append([event_test['X'], event_test['Y'], event_test['Z'],
					 event_test['t']])

		if i > p3p:
			if (patience%i) == 0:
				# In cases where the convergence threshold is reached before
				# the n-th iterations are made, if the slope is sufficiently
				# low then we break the loop.
				if (np.mean(cost_story[-p2p:-patience])-
								np.mean(cost_story[-patience:])==0):
					break

	cost_story = np.array(cost_story)
	history = np.array(history, dtype=object)
	misfit_fin = calc_misfit(stations, event_test, vp)
	# print("Best event = {}, best misfit = {:.10f}".format(event_test,
	# 														misfit_fin))

	return event_test, cost_story, history
