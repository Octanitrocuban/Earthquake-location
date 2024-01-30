# Earthquake-location
This repository contain different method to locate earthquake in a higly simplified environment. 

### There are 3 scripts:
**gradient_descent.py**: this module contains function to compute gradient descent through exploration algorithms.
  - centrering: function to normalise (centralise) stations and event if given.
  - calc_misfit: function to calculate misfit between observed and calculated arrival times.
  - descente_gradient: function to make a gradient descent from a given test event and data from stations.

**graph.py**: this module contain functions to plot the state of the search and the history of the localisation of the earthquake.
  - plot_dict_stations: function to plot the position of the station and of the tested earthquake(s) if given.
  - plot_history_dict: function to plot the evolution of the variables.
  - plot_diff_cost_evol: function to plot the change in variation of the rmse through the epochs.

**example.py**: this script contain exemples on how to use the functions from gradient_descent.py and graph.py.

### Plots:
Here are examples and statistics of descente_gradient function:



