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
#=============================================================================
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

# compute data to be relative to a reference station
Stations, Event = gd.centrering(Stations, 'S4', Event)
print('True values:', Event, '\n')

#==================
#==================
# Method using a random starting set of parameters to make a gradient descent
print('Descente from one samples')
Supp  = {'X':np.random.uniform(-.5, .5)*fact,
		 'Y':np.random.uniform(-.5, .5)*fact,
		 'Z':-500, 't':-0.5}

event_test, Cost_story, Hist = gd.descente_gradient(Stations, Supp,
												    1000000, l_r_m=0.2,
													patience=10000)


gh.plot_history_dict(Stations, event_test, Hist, Cost_story, Event, 100)
