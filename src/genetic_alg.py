# -*- coding: utf-8 -*-
"""
This module contain functions to compute gradient descent to localise the
earthquake through a genetic algorithm.
"""
import numpy as np
from scipy.spatial.distance import cdist
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

def rmse(stations, events, vp=4000):
	"""
	Calculate misfit between observed and calculated arrival times using
	numpy ndarrays.

	stations : numpy.ndarray
		A 2d numpy array recording the stations data.
	events : numpy.ndarray
		A 2d numpy array recording the data of the tested events.
		It have the shape : ([[X0, Y0, Z0, t0], ..., [Xn, Yn, Zn, tn]])
	vp : float
		Value of the P-wave velocity. It is assumed constant.

	Returns
	-------
	misfit : numpy.ndarray
		Root Mean Square Error between the data recorded by the stations
		and the tested events.

	"""
	dist = cdist(stations[:, :3], events[:, :3]).T
	tp_est = events[:, 3:4] + dist / vp
	misfit = np.sum((tp_est - stations[:, 3])**2, axis=1)**.5 /len(stations)
	return misfit

def randoms(shape, limits_u=None, limits_n=None):
	"""
	Function to create random samples with a given type of pdf and
	parameters.

	Parameters
	----------
	shape : tuple
		Shape of the new samples randomly created.
	limits_u : numpy.ndarray, optional
		Limits of the space to explore. The shape will be (n, 2) with n the
		number of parameters. For the n-th line, it is the (lower, higgher)
		bound. The default is None.
	limits_n : numpy.ndarray, optional
		Limits of the space to explore. The shape will be (n, 2) with n the
		number of parameters. For the n-th line, it is the (mean, deviation).
		The default is None.

	Returns
	-------
	news : numpy.ndarray
		Listing of the new samples.

	"""
	if type(limits_u) == np.ndarray:
		news = np.random.uniform(0, 1, shape)
		news = news * (limits_u[:, 1]-limits_u[:, 0]) + limits_u[:, 0]

	elif type(limits_n) == np.ndarray:
		news = np.random.normal(0, 1, shape)
		news = news * limits_n[:, 1] + limits_n[:, 0]

	else:
		raise TypeError('Either limits_u or limits_n must be numpy array.')

	return news

def selections(score, population, propor):
	"""
	Function to make selectionof the bests samples for givens scores, samples
	and the number to be kept.

	Parameters
	----------
	score : numpy.ndarray
		RMSE between station's data and the proposed samples.
	population : numpy.ndarray
		Proposed samples.
	propor : int
		Number of sample to keep (number of survivors).

	Returns
	-------
	survivors : numpy.ndarray
		Samples kept.
	rank : numpy.ndarray
		Indices of the sorted best samples.

	"""
	rank = np.argsort(score)[:propor]
	survivors = population[rank]
	return survivors, rank

def means(survivors, length):
	"""
	Function to radomly create couple of samples and retrun their average
	value.

	Parameters
	----------
	survivors : numpy.ndarray
		Samples kept from selections function.
	length : int
		Twice the number of samples that were kept.

	Returns
	-------
	news : numpy.ndarray
		The averaged couple of samples.

	"""
	indices = np.zeros(length, dtype=int)
	indices[np.random.choice(np.arange(length), length//2, False)] = 1
	news = (survivors[indices == 0]+survivors[indices == 1])/2
	return news

def cross(survivors, length, combinaisons, ncb=6):
	"""
	Function to radomly create couple of samples and retrun their random mix
	values.

	Parameters
	----------
	survivors : numpy.ndarray
		Samples kept from selections function.
	length : int
		Number of samples that were kept.
	combinaisons : numpy.ndarray
		A 2d array storing the combinaisons.
	ncb : int
		Number of combinaisons.

	Returns
	-------
	news : numpy.ndarray
		The cross-selection between the couple of samples.

	"""
	index = combinaisons[np.random.choice(np.arange(ncb), length, True)]
	indices = np.zeros(length*2, dtype=int)
	indices[np.random.choice(np.arange(length), length, False)] = 1
	news = np.zeros((length, 4))
	news[index == 0] = survivors[indices == 0][index == 0]
	news[index == 1] = survivors[indices == 1][index == 1]
	return news

def mutants(survivors, shape, rate):
	"""
	Function to create mutants samples from the survivor samples.

	Parameters
	----------
	survivors : numpy.ndarray
		Samples kept from selections function.
	shape : tuple
		Shape of survivor array.
	rate : numpy.ndarray
		A 1d array weighting the level of noise added to the muatant samples.

	Returns
	-------
	news : numpy.ndarray
		The mutated samples.

	"""
	news = survivors+np.random.normal(0, 1, shape)*rate
	return news

def evolution(size, n_g, noise, limites, stations, p_surv, combinaison,
			  rate, treshold, patience=100):
	"""
	Function to search a sample with the better parameters possible through
	genetic algorithm.

	Parameters
	----------
	size : int
		Length of the full population.
	n_g : int
		Number of generation.
	noise : str
		Type of noise used in the random samples. If it is 'normal' it will
		draw samples from a multi-normal law with the means beeing the first
		column of limites array, and the standard deviations beeing the
		second column of limites array. If it is 'uniform' it will draw
		samples from a multi-uniform law with the lower boundaries beeing the
		first column of limites array, and the upper boundaries beeing the
		second column of limites array.
	limites : numpy.ndarray
		Parameters to parametrise the probability law we will sample.
	stations : numpy.ndarray
		A 2d numpy array recording the stations data.
	p_surv : list
		List of int: [Number of survivor after selection, Number of survivor
		after selection dived by 2, Number of random samples to have the
		right number of samples]. Given n the number of sample in total and
		alpha the prortion of them to survive, their will be n*alpha
		survivor, n*alpha/2 means, n*alpha/2 cross-selected, n*alpha mutants
		and n-3*n*alpha random samples. This implie the alpha is < 1/3.
	combinaison : numpy.ndarray
		A 2d array storing the combinaisons.
	rate : numpy.ndarray
		A 1d array weighting the level of noise added to the muatant samples.
	treshold : float
		Minimum value wich if reach will lead to early stoping.
	patience : int, optional
		Patience for learning rate reduction. The default is 100.

	Returns
	-------
	best : numpy.ndarray
		History of the best event (with the lowest loss).
	loss : numpy.ndarray
		Loss history of the best event (lowest loss) at each iteration.

	"""
	p2p = int(2*patience)
	upper = int(3*patience)
	if noise == 'normal':
		population = randoms((size, 4), limits_n=limites)
	elif noise == 'uniform':
		population = randoms((size, 4), limits_u=limites)

	score = rmse(stations, population)
	print('Starting misfit = '+str(np.min(score)))
	loss = []
	best = []
	for i in range(1, n_g+1):
		score = rmse(stations, population)
		survivants, rank = selections(score, population, p_surv[0])
		loss.append(score[rank][0])
		best.append(survivants[0])
		if (i > upper)&((i%patience) == 0):
			if np.min(loss[-p2p:-patience]) <= np.min(loss[-patience:]):
				rate *= 0.9

		if loss[-1] <= treshold:
			break
		else:
			moyens = means(survivants, p_surv[0])
			crossover = cross(survivants, p_surv[1], combinaison)
			mutors = mutants(survivants, (p_surv[0], 4), rate)
			if noise == 'normal':
				rands = randoms((p_surv[2], 4), limits_n=limites)
			elif noise == 'uniform':
				rands = randoms((p_surv[2], 4), limits_u=limites)

			population = np.concatenate((survivants, moyens, crossover,
										 mutors, rands))

	loss = np.array(loss)
	print("Best misfit = "+str(np.min(loss)))
	best = np.array(best)
	last = np.argwhere(loss == np.min(loss))[-1]
	print_sample_array(best[last[-1]], before='Best event = ')
	return best, loss
