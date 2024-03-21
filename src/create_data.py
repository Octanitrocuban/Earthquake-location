'''
Script to create synthetic data to train machine learning models.
'''
import numpy as np
from scipy.spatial.distance import cdist
#=============================================================================

# FUNCTIONS
def compute_box(box_limites, num_boxes):
	"""
	Function to compute the center of the cells of a 3d box.

	Parameters
	----------
	box_limites : numpy.ndarray
		Upper and lower limites of the axis of the box.
	num_boxes : numpy.ndarray
		Number of sub-box per axis.

	Returns
	-------
	boxes : numpy.ndarray
		Centers of the cells of the box.

	"""
	center_x = np.linspace(box_limites[0, 0], box_limites[0, 1], num_boxes+1)
	center_y = np.linspace(box_limites[1, 0], box_limites[1, 1], num_boxes+1)
	center_z = np.linspace(box_limites[2, 0], box_limites[2, 1], num_boxes+1)
	center_x = (center_x[1:]+center_x[:-1])/2
	center_y = (center_y[1:]+center_y[:-1])/2
	center_z = (center_z[1:]+center_z[:-1])/2
	boxes = np.meshgrid(center_x, center_y, center_z)
	boxes = np.array([np.ravel(boxes[0]), np.ravel(boxes[1]),
					  np.ravel(boxes[2])]).T

	return boxes

def noise_box(X, multiplicator, differences):
	"""
	Function to compute random new positions for each X i-th positions within
	given boundaries.

	Parameters
	----------
	X : numpy.ndarray
		The initial positions.
	multiplicator : int
		Number of new positions per initial position.
	differences : numpy.ndarray
		Width per axis that are aroud the initial positions.

	Returns
	-------
	noise : numpy.ndarray
		Random values that will be add to the initial positions.

	"""
	noise = np.random.uniform(-1, 1, 
							  (X.shape[0], multiplicator,
								 X.shape[1]))*differences

	return noise

def uppsample_box(centers, multiplicator, differences):
	"""
	Function to compute a random sampling of the cells of the box.

	Parameters
	----------
	centers : numpy.ndarray
		Centers of the cells of the box.
	multiplicator : int
		Number of new positions per center.
	differences : numpy.ndarray
		Width per axis that are aroud the center of the cells.

	Returns
	-------
	uppped : numpy.ndarray
		The randomly uppsampled positions of the box.

	"""
	noise = noise_box(centers, multiplicator, differences)
	uppped = centers[:, np.newaxis]+noise
	uppped = np.reshape(uppped, (uppped.shape[0]*multiplicator, 3))
	return uppped

def compute_timing(stations, samples, vp=4000, level=0.0001):
	"""
	Function to compute the
	noise that will be add to the arrival time of the stations.
	

	Parameters
	----------
	stations : numpy.ndarray
		Array with the x, y and z position of the stations.
	samples : numpy.ndarray
		X, y and z position of the earthquakes.
	vp : float, optional
		P-wave celerity in the medium.
	level : float, optional
		A weight to have more or less noisy data.

	Returns
	-------
	timing : numpy.ndarray
		Time of the P-wave propagation between the earthquakes and the
		stations with noise. The noise is proportional to the distance
		between the earthquakes and the stations

	Note
	----
	The noise is proportional to the distance between theearthquakes and
	the stations with a weight factor (level). The more `level` will be
	important, the more time will be noisy. If `level` = 0, then no noise
	will be add.

	"""
	timing = cdist(stations[:, :3], samples)/vp
	timing_noise = np.random.uniform(-1, 1, timing.shape)*timing*level
	timing = timing+timing_noise
	return timing

def XY_normalization_X(X, maximums, nfeatures):
	"""
	Function to normalize the x and y position of the stations in the X
	matrix.

	Parameters
	----------
	X : numpy.ndarray
		Matrix that will be used as input for the model.
	maximums : numpy.ndarray
		A 1d array. It is computed as follow:
		np.max(stations[:, :-1]-np.min(stations[:, :-1], axis=0), axis=0)
	nfeatures : int
		Number of features per stations. It is 4 if we have x, y, z and the
		relative arrival time.

	Returns
	-------
	X : numpy.ndarray
		Matrix that will be used as input for the model.

	"""
	kernel_x = np.arange(0, X.shape[1], nfeatures)
	kernel_y = np.arange(1, X.shape[1]+1, nfeatures)
	X[:, kernel_x] = X[:, kernel_x]/maximums[0]
	X[:, kernel_y] = X[:, kernel_y]/maximums[1]
	return X

def XY_normalization_y(y, limites):
	"""
	Function to 

	Parameters
	----------
	y : numpy.ndarray
		Matrix that will be used as output for the model.
	limites : numpy.ndarray
		Upper and lower limites of the axis used for the box creation.

	Returns
	-------
	y : numpy.ndarray
		Matrix that will be used as output for the model.

	"""
	y = (y-limites[:, 0])/(limites[:, 1]-limites[:, 0])
	y[:, -1] = y[:, -1]-1
	return y

def centering_X(X, reference, n_features):
	"""
	Function to transform the time between the stations and the earthquake
	to the relative arrival time.

	Parameters
	----------
	X : numpy.ndarray
		Matrix that will be used as input for the model.
	reference : int
		Index of the station used as reference. It usually the station closet
		to the average position of the stations.
	n_features : int
		Number of features per stations. It is 4 if we have x, y, z and the
		relative arrival time.

	Returns
	-------
	X_center : numpy.ndarray
		Matrix that will be used as input for the model.

	"""
	start = int(reference*n_features)
	stop = int((reference+1)*n_features)
	n_stations = X.shape[1]//n_features
	center = []
	for i in range(n_stations):
		center.append(np.copy(X[:, start:stop]))

	center = np.hstack(tuple(center))
	X_center = X-center
	return X_center


# PARAMETERS

# Weight on distances of the stations
fact = 1000

# Number of box per dimensions
n_boxes = 20

# Number of new samples per box
mult = 10

# repartition of the boxs between the sets
train_p = 0.7
valid_p = 0.15
test_p = 1-valid_p-train_p

# P-wave celerity
vp = 4000 # m/s

# noise level on the arrival time
noise_lv = 0.0001

# folder to save the data
folder = '../data/'


# VERIFICATIONS

if fact == 0:
	raise ValueError('The weight factor of the positions of the stations '+
					 'cannot be equal to three. Found `n_boxes = '+
					 str(n_boxes)+'`')

if n_boxes <= 3:
	raise ValueError('The number of boxes per axis cannot be lower or equal '+
					 'than three. Found `n_boxes = '+str(n_boxes)+'`')

if mult <= 0:
	raise ValueError('The number of sample per box cannot be lower or equal '+
					 'than zeros. Found `mult = '+str(mult)+'`')

if test_p <= 0:
	raise ValueError('The size of the test set cannot be lower or equal than '+
					 'zeros. Found `test_p = '+str(test_p)+'`')

if vp <= 0:
	raise ValueError('P-wave celerity cannot be lower or equal than zeros. '+
					 'Found `vp = '+str(vp)+'`')

if noise_lv < 0:
	raise ValueError('The level of noise on the arrival time cannot be lower '+
					 'than zeros. Found `noise_lv = '+str(noise_lv)+'`')


# CODE

# Stations
Stations = np.array([[0.0, 0.7, 0.0, 0.0],
					 [0.7, 1.0, 0.0, 0.0],
					 [1.0, 0.3, 0.0, 0.0],
					 [0.5, 0.5, 0.0, 0.0],
					 [0.4, 0.0, 0.0, 0.0]]) * fact

# Number of stations
n_s = len(Stations)

# Compute data to be relative to a reference station
ref_sta = np.copy(Stations[3])
Stations = Stations-ref_sta

# Create the data for the station
# Maximum distance
max_d = (3*fact**2)**0.5

# Maximum time offset
max_t = -(max_d/vp+max_d/vp*noise_lv)

# Splitting the space into subset that will be used for train, validation
# and test:
limits = np.array([[np.min(Stations[:, 0]), np.max(Stations[:, 0])],
					[np.min(Stations[:, 1]), np.max(Stations[:, 1])],
					[-fact, 0]])

# Steps to have noise centered in the boxes
differ = ((limits[:, 1]-limits[:, 0])/(n_boxes*2))

# Compute 3d grid of the center of the bosex
boxs = compute_box(limits, n_boxes)
length = boxs.shape[0]

# repartition of the boxs between the sets:
prop = [int(train_p*length), int(valid_p*length)]

# label 0:train, 1:valid, 2:test
sets = np.zeros(length, dtype=int)
sets[prop[0]:] = 1
sets[prop[0]+prop[1]:] = 2
np.random.shuffle(sets)

# Splitting
train = boxs[sets == 0]
valid = boxs[sets == 1]
test = boxs[sets == 2]

# Samples
rand_train = uppsample_box(train, mult, differ)
rand_valid = uppsample_box(valid, mult, differ)
rand_test = uppsample_box(test, mult, differ)

# x, y, z to t
y_train = np.copy(rand_train)
y_valid = np.copy(rand_valid)
y_test = np.copy(rand_test)

# Computing the time with noise
train_t = compute_timing(Stations, y_train, vp=vp, level=noise_lv)
valid_t = compute_timing(Stations, y_valid, vp=vp, level=noise_lv)
test_t = compute_timing(Stations, y_test, vp=vp, level=noise_lv)

# initialization
X_train = np.zeros((train_t.shape[1], n_s*4))
X_valid = np.zeros((valid_t.shape[1], n_s*4))
X_test = np.zeros((test_t.shape[1], n_s*4))

# Loop to fill the arrays with the positions of the stations and the noised
# time took by P-wave to reach them
for i in range(n_s):
	X_train[:, i*4:(i+1)*4-3] = Stations[i, 0]
	X_train[:, i*4+1:(i+1)*4-2] = Stations[i, 1]
	X_train[:, i*4+2:(i+1)*4-1] = Stations[i, 2]
	X_train[:, i*4+3:(i+1)*4] = train_t[i, :, np.newaxis]

	X_valid[:, i*4:(i+1)*4-3] = Stations[i, 0]
	X_valid[:, i*4+1:(i+1)*4-2] = Stations[i, 1]
	X_valid[:, i*4+2:(i+1)*4-1] = Stations[i, 2]
	X_valid[:, i*4+3:(i+1)*4] = valid_t[i, :, np.newaxis]

	X_test[:, i*4:(i+1)*4-3] = Stations[i, 0]
	X_test[:, i*4+1:(i+1)*4-2] = Stations[i, 1]
	X_test[:, i*4+2:(i+1)*4-1] = Stations[i, 2]
	X_test[:, i*4+3:(i+1)*4] = test_t[i, :, np.newaxis]

# For saving raw values
X_train_raw = np.copy(X_train)
X_valid_raw = np.copy(X_valid)
X_test_raw = np.copy(X_test)
y_train_raw = np.copy(y_train)
y_valid_raw = np.copy(y_valid)
y_test_raw = np.copy(y_test)

# Centering the X_raw data
X_train_raw = centering_X(X_train_raw, 3, 4)
X_valid_raw = centering_X(X_valid_raw, 3, 4)
X_test_raw = centering_X(X_test_raw, 3, 4)

# for normalisation
min_X = np.min(Stations[:, :-1], axis=0)
max_X = np.max(Stations[:, :-1]-min_X, axis=0)

# for X we only normalise the position of the stations
X_train = XY_normalization_X(X_train, max_X, 4)
X_valid = XY_normalization_X(X_valid, max_X, 4)
X_test = XY_normalization_X(X_test, max_X, 4)

# for y we only normalise the position of the earthquakes
y_train = XY_normalization_y(y_train, limits)
y_valid = XY_normalization_y(y_valid, limits)
y_test = XY_normalization_y(y_test, limits)

# Centering the X_normalised data
X_train = centering_X(X_train, 3, 4)
X_valid = centering_X(X_valid, 3, 4)
X_test = centering_X(X_test, 3, 4)

# Saving the parameters used to create the datasets
np.save(folder+'parameters.npy', np.array([{
		'min_X':min_X,
		'max_X':max_X,
		'stations_cent':Stations,
		'factor':fact,
		'n_boxes':n_boxes,
		'mult':mult,
		'train_p':train_p,
		'valid_p':valid_p,
		'test_p':test_p,
		'vp':vp,
		'noise_lv':noise_lv,
		'nfeatures':4,
		'ref_station':3,
		'X_norm':'X[:, kernel_x|y]/min_X[0|1]',
		'limites':limits,
		'y_norm':('(y-limites[:, 0])/(limites[:, 1]-limites[:, 0])',
					'y[:, -1] = y[:, -1]-1'),
		'differences':differ}]))

# Saving the normalised datasets
np.save(folder+'train_norm.npy', np.array([{'X_train':X_train,
											 'y_train':y_train}]))

np.save(folder+'valid_norm.npy', np.array([{'X_valid':X_valid,
											 'y_valid':y_valid}]))

np.save(folder+'test_norm.npy', np.array([{'X_test':X_test,
											'y_test':y_test}]))

# Saving the raw datasets
np.save(folder+'train_raw.npy', np.array([{'X_train_raw':X_train_raw,
											'y_train_raw':y_train_raw}]))

np.save(folder+'valid_raw.npy', np.array([{'X_valid_raw':X_valid_raw,
											'y_valid_raw':y_valid_raw}]))

np.save(folder+'test_raw.npy', np.array([{'X_test_raw':X_test_raw,
										   'y_test_raw':y_test_raw}]))
