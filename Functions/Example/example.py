# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:29:14 2023

@author: etocc
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from scipy import signal
import sys
import time
import pickle

sys.path.append('C:/Users/etocc/Documents/Multi-site project/Multi-site_LFP/Functions')
import Processsing_functions as pf

#Define a function to plot trajectories
def plot_traject(x,y):
    plt.plot(x,y)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

os.chdir('C:/Users/etocc/Documents/Multi-site project/Multi-site_LFP/Functions/Example')
#Load dictionary of events
a_file = open("feeding.pickle", "rb")
events = pickle.load(a_file)


#Load dataframe of centroids positions
centroids_df=pd.read_csv('Centroids.csv', index_col=0)

#Plot trajectory
plot_traject(centroids_df['Centroid1_X'],centroids_df['Centroid1_Y'])

#It looks like there is is some noise, in partictular that the camera failed to detect the mouse, which was recorded as 
# position at (0,0) coordinate, as well as it detected some extreme values at "the right", this means, extreme X values 
#(probable the reflect of the mouse in the waal of the chamber). Let's do some cleaning and plot again...
centroids_df.replace(0,np.nan, inplace=True)
centroids_df.dropna(inplace=True)
centroids_df=centroids_df.loc[(centroids_df['Centroid1_X']<=np.quantile(centroids_df['Centroid1_X'], 0.999))&(centroids_df['Centroid1_Y']<=np.quantile(centroids_df['Centroid1_Y'], 0.999))]

plot_traject(centroids_df['Centroid1_X'],centroids_df['Centroid1_Y'])


# Much better now. Let's scale values using scale_centroids function. The ceasures of the chamber were 17 cm x 32 cm
# Since the input must be an numpy array of shape (2 x n), we need to get the values of the dataframe, and transpose them.
scaled_centroids=pd.DataFrame(pf.scale_centroids(centroids_df.values.T,17,30).T, index=centroids_df.index, columns=['Centroid_X (cm)','Centroid_Y (cm)'])
plot_traject(scaled_centroids['Centroid_X (cm)'],scaled_centroids['Centroid_Y (cm)'])

# Calculate speed using calculate_speed function and add it to the dataframe as a new column
scaled_centroids['Speed']=pf.calculate_speed(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),1)


## Create movement stop event, this will be a function soon!
from scipy.ndimage import label, find_objects # import neccesary functions from scipy
rest=np.where(scaled_centroids['Speed'].values<0.5,1,0) #Identify resting periods based using a treshold of speed=0.5 cm/seg
labelled_rest,number_of_feaures=label(rest) # label every period of consecutive rest with a nuumber
slices=find_objects(labelled_rest) # make slices for every period
long_slices=[i for i in slices if labelled_rest[i].size>450] #filter for periods longer than 5 seconds 
start_index=[i[0].start for i in long_slices] #sample long_slices
rest_starts=np.array(scaled_centroids.reset_index().loc[start_index, 'index'])

#Add rest starts to events dictionary
events.update({'rest_starts':rest_starts})

#Plot the positions of rest start and feeding
for event in events.keys():
    pf.position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events[event],plot=event)

    
#It looks like a large amount of rests occur besides the feeding zone. Let's look what happens with speed when animals eat
pf.perievent_histogram(scaled_centroids[['Speed']], events, -20, 20, number_of_decimals=2)


#Since mice stop moving to eat, feeding may be being confused with resting. By filtering the resting starts that occur
# temporally close from feeding events, we should be able to solve this issue. This can be achieved with the clasify_events_base_on_time function
false_rest_starts, real_rest_starts=pf.clasify_events_base_on_time(rest_starts, events['feeding'],10,mode='booth')

#Remove rest starts from events dictionary, and add real rest starts and false rest starts
del(events['rest_starts'])
events.update({'real_rest_starts':real_rest_starts,'false_rest_starts':false_rest_starts})

#Plot positions and peri-event histograms of speed again to verify the processing worked
for event in events.keys():
    pf.position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events[event],plot=event)

    
pf.perievent_histogram(scaled_centroids[['Speed']], events, -20, 20, number_of_decimals=2)

#Make distance to feeding column using position_of_event function
x,y,distances=pf.position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events[event], plot=False)
scaled_centroids['Distance to feeding']=distances
#Plot peri-event histogram of distance to feeding locked to every event, to verify that the filtering worked
pf.perievent_histogram(scaled_centroids[['Distance to feeding']], events, -20, 20, number_of_decimals=2)



