# -*- coding: utf-8 -*-
"""
This script contain exemples on how to use the functions from
gradient_descent.py and graph.py.
"""
#import usefull library
import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd
import graph as gh
from time import time
from tqdm import tqdm
#=============================================================================
to_do = ['monte_carlo']

# Create the stations
fact = 1000
# dict structure
Stations = {'S1':{'X':0,		'Y':0.7*fact, 'Z':0},
			'S2':{'X':0.7*fact, 'Y':1*fact,   'Z':0},
			'S3':{'X':1*fact,   'Y':0.3*fact, 'Z':0},
			'S4':{'X':0.5*fact, 'Y':0.5*fact, 'Z':0},
			'S5':{'X':0.4*fact, 'Y':0,		  'Z':0}}

# Create an event
Event = {'X': np.random.uniform(0, 1*fact),
		 'Y': np.random.uniform(0, 1*fact),
		 'Z': -np.random.uniform(0, 1*fact),
		 't': float(time())}

# array structure
sta_a = np.array([[0.0, 0.7, 0.0, 0.0],
				  [0.7, 1.0, 0.0, 0.0],
				  [1.0, 0.3, 0.0, 0.0],
				  [0.5, 0.5, 0.0, 0.0],
				  [0.4, 0.0, 0.0, 0.0]]) * fact

# array structure
Event_a = np.array([Event['X'], Event['Y'], Event['Z'], Event['t']])

# create the data for the station
vp = 4000 # m/s
# dict structure
for i in Stations:
	# distances
	dist = ((Event['X']-Stations[i]['X'])**2
			+(Event['Y']-Stations[i]['Y'])**2
			+(Event['Z']-Stations[i]['Z'])**2)**0.5

	# recorded time + noise
	noise = np.random.uniform(-1, 1)*dist/vp*0.0001
	t = Event['t'] + dist/vp + noise
	Stations[i]['t'] = t

# array structure
for i, s in enumerate(Stations):
	sta_a[i, 3] = Stations[s]['t']

# compute data to be relative to a reference station
Stations, Event = gd.centrering(Stations, 'S4', Event)
sta_a, Event_a = gd.centrering(sta_a, 3, Event_a)
print('True values:', Event, '\n')

#==================
#==================
# Method using a random starting set of parameters to make a gradient descent
if 'single_descent' in to_do:
	print('Descent from one samples')

	Supp  = {'X':np.random.uniform(-.5, .5)*fact,
			 'Y':np.random.uniform(-.5, .5)*fact,
			 'Z':-500, 't':-0.5}

	event_test, hist, cost_story = gd.gradient_descent(Stations, Supp,
													   100000, l_r_m=0.2,
													   patience=10000)

	gh.plot_history_dict(Stations, event_test, hist, cost_story, Event, 100)
	print('')

#==================
#==================
# Method to use multiple random starting set of parameters to make a gradient
# descent
if 'ensemble_descent' in to_do:
	print('Descent from multiple samples')
	limits = [[np.min(sta_a[:, 0]), np.max(sta_a[:, 0])],
			  [np.min(sta_a[:, 1]), np.max(sta_a[:, 1])],
			  [-1000., 0.],
			  [   -1., 0.]]

	# to not adjust a parameter, set its learning rate to 0
	lrm = np.array([1., 1., 0.1, 0.0001])
	n_samp = 500
	trained, erreur = gd.ensemble_descent(sta_a, n_samp, lrm, 10000, vp,
										  limits, patience=1000)

	gh.plot_history_vect_ed(sta_a, trained, erreur, Event_a, 500)
	print('')


#==================
#==================
# Method to use Monte Carlo sampling of the parameters space
if 'monte_carlo' in to_do:
	print('Monte Carlo')
	length = int(8e6)
	limits = np.array([[ -500, 500], [ -500, 500],
					   [-1000,   0], [   -1,   0]])

	samples, losses = gd.monte_carlo(length, limits, sta_a, 'random')
	best = samples[np.argmin(losses)]
	gh.plot_vect_stations(sta_a, Event_a, best[np.newaxis])
	print('')


#==================
#==================
# Method to use deepening grid search
if 'deepen_grid' in to_do:
	print('Deepening grid search')
	n = 10
	depth = 100
	wid_f = 4.
	limits = np.array([[ -500, 500], [ -500, 500],
					   [-1000,   0], [   -1,   0]])

	sample_story, cost_story = gd.deepening_grid_search(sta_a, n, limits, depth, wid_f)
	gh.plot_history_vect(sta_a, s_his[-1], sample_story, cost_story, Event_a, 1)
	print('')

