# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 08:56:19 2023

@author: etocc
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import signal
import time

# from mne_connectivity import spectral_connectivity_epochs




#Define functions


def peri_event_time(continuous, events, lower_limit, higher_limit, number_of_decimals=3, all_trials=False):
    """
    Make a column with peri-event times, using the formula time-event_time, for time value from lower_limit to higher limit.

    Parameters
    ----------
    continuous : pandas.DataFrame
        Dataframe with continuous variables (LFP, speed, position) in each column, and timestamps as index.
    events : dict or pandas.DataFrame
        Timestamps of each event.If a dict, keys must be events name as str and values must be timestamps of each event as list, numpy.array or pandas.Serie;
        if a pandas.DataFrame, it must be a column named "Event" with event's names, and timestamps as index.
    lower_limit : float
        Lower limit of the time interval.
    higher_limit : float
        Higher limit of the time interval.
    number_of_decimals : int, optional
        Number of decimals of the peri-event time; it should be the same as the number of decimals in the index of continuous paramater. The default is 3.
    all_trials : bool, optional
        If True, it returns a dataframe with all trials of each event, and adds "Event_number" column, containing counts of each event. If it is False,
        it averages trials of the same event timestamp-wise, and returns a dataframe with averages per event. The default is False.

    Raises
    ------
    TypeError
        If continuous type is not pandas.DataFrame or if events type is neither dict or pandas.DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        DataFRame with original continous columns, plus an "Event" column with event names, and a "Peri-event_time" column with peri-event timestamps.
        Timestamps outside of higher and lower limits are not included, and index doe not have meaningful information other than row id.
        Read "all_trials" description above for differences in the returned DataFrame if aa_trials is True or False.

    """
    
    if type(events)==dict:
        events_df=pd.DataFrame({key:pd.Series(value) for key,value in events.items()}).melt(var_name='Event', value_name='Timestamp').dropna().set_index('Timestamp').sort_index()
        events_df.index=np.round(event_df.index, 3)
    elif type(events)==pd.DataFrame:
        events_df=events.dropna().sort_index()
    else:
        raise TypeError ('"events" must be either a dictionary with keys="Event names":values=list of timestamps, or a pandas.DataFrame with "Event" column corresponding to event name, and timestamps as index')
    if type(continuous) != pd.DataFrame:
        raise TypeError ('"continuous" must be a pandas.DataFrame with continuous values as columns and timestamps as index')
    
    df_list=[]
    if all_trials==True:
        for event in events_df['Event'].unique():
            event_count=0
            df=continuous.copy()
            for i in events_df[events_df['Event']==event].index:
                df.loc[i+lower_limit:i+higher_limit, 'Peri-event_time']=np.round(df.loc[i+lower_limit:i+higher_limit, ].index-i, number_of_decimals)
                df.loc[i+lower_limit:i+higher_limit, 'Event']=event
                df.loc[i+lower_limit:i+higher_limit, 'Event_number']=event_count
                event_count+=1
            df.dropna(subset=['Peri-event_time'], inplace=True)
            df_list.append(df)
        df=pd.concat(df_list, ignore_index=True)
    else:
        for event in events_df['Event'].unique():
            df=continuous.copy()
            for i in events_df[events_df['Event']==event].index:
                df.loc[i+lower_limit:i+higher_limit, 'Peri-event_time']=np.round(df.loc[i+lower_limit:i+higher_limit, ].index-i, number_of_decimals)
                df.loc[i+lower_limit:i+higher_limit, 'Event']=event
            df.dropna(subset=['Peri-event_time'], inplace=True)
            df_list.append(df)
        df=pd.concat(df_list, ignore_index=True)
        df=df.groupby(['Event','Peri-event_time'], as_index=False).mean()
    return df
            
  
def scale_centroids(centroids, x_dim,y_dim):
    """
    Scale centroids to match the real values in appropiate scale. It assumes position at lower and higher limits of each dimension have been recorded.

    Parameters
    ----------
    centroids : numpy.ndarray of shape (2 x n_timestamps)
        Values of X_centroid and Y_centroid to be scaled.
    x_dim : float
        Value from end to end of X coordinate.
    y_dim : float
        Value from end to end of Y coordinate.

    Returns
    -------
    numpy.ndarray with scaled values.

    """
    x_min, x_max=centroids[0].min(),centroids[0].max()
    y_min, y_max=centroids[1].min(),centroids[1].max()
    x_dist=x_max-x_min
    y_dist=y_max-y_min
      
    scaled_dif=(x_dist/y_dist - x_dim/y_dim)/(x_dim/y_dim)
    if abs(scaled_dif) > 0.1:
        print('\nWarning: the observed and provided x/y ratios have a difference of above 10%, consider switching the order of X and Y dimensions or cleaning centroids data and scale again')
    
    print('\nDifference between observed and provided x/y ratios:',100*scaled_dif, '%')
    
    x_multiplying_factor=x_dim/x_dist
    y_multiplying_factor=y_dim/y_dist
    min_values=np.array([x_min,y_min]).reshape(2,1)
    multiplying_factors=np.array([x_multiplying_factor,y_multiplying_factor]).reshape(2,1)
    result=(centroids-min_values)*multiplying_factors
    return(result)

def calculate_speed(X_values, Y_values, timestamps, smooth=0):
    '''
    Calculate scalar speed through the following steps:
    1. Calculate dtime, dX=PosX[T-deltaT]-posX[T] and dY=PosY[T-deltaT]-posY[T]
    2. Smooth dX and dY using a Gaussian filter. Note: by applying the Gaussian filter to dX and dY instead of the speed
    avoids short movements or random noie of the camera  to propagate to the speed calculus, leaving only changes in position caused 
    by displacement
    3. Calculate scalar speed: sqrt(dX**2+dY**2) / dT

    Parameters
    ----------
    X_values : NUMPY ARRAY
        Positions in X coordinate. Must have the same lenght than Y_values and timestamps
    Y_values :  NUMPY ARRAY
        Positions in Y coordinate. Must have the same lenght than X_values and timestamps
    timestamps : NUMPY ARRAY
        Timestamps for every position. Must have the same lenght than X_values and Y_values
    smooth : Float, optional
        Defines the window of time a 2 x smooth and the width of the  Gaussian kernel as 1 x smooth,
        for the Gaussian filter. The default is 0, in which case the Gaussian filter is not appplied.

    Returns
    -------
    speed: numpy.ndarray
        Values of scalar speed

    '''
    #Evaluate if X_values, Y_values and timestamps ahve the same lenght, raise error if not
    if len(X_values)!=len(Y_values) or len(X_values)!=len(timestamps):
        raise ValueError ('X_values, Y_values and timestamps must have the same lenght')
    #Evaluate if X_values, Y_values and timestamps are of type np.array, raise error if not
    if type(X_values)!=np.ndarray or type(Y_values)!=np.ndarray or type(timestamps)!=np.ndarray:
        raise TypeError ('X_values, Y_values and timestamps must be of type numpy.ndarray')
    #Evaluate if smooth is of type int or float, raise error if not
    if type(smooth)!= float and type(smooth)!= int:
        raise TypeError('smooth must be integer or float')
    
    #Calculate nstantaneus change in time
    # dif_time=np.round(np.array((scaled_centroids.index-np.roll(scaled_centroids.index, 1))),3)
    dT=np.round(np.roll(timestamps, -1)-timestamps,3)
    dT[-1]=np.nan
    #Calculate average dif time, which will be used for the smoothing process
    avg_dT=np.mean(dT[0:-1])
    #Calculate the number of rows to use a std in the gaussian filter, by dividing the smooth time provided by the average delta time
    smooth2=smooth/avg_dT
    #Calculate the differential X and Y, by substracting each value to the next value (using shift)   
    dX=np.roll(X_values,-1)-X_values
    dX[-1]=np.nan #last value is wrong, because np.roll put the last value at firt position
    dY=np.roll(Y_values,-1)-Y_values
    dY[-1]=np.nan #last value is wrong, because np.roll put the last value at firt position
    if smooth2!=0:  # Only apply smoothing if smooth2 is different from 0  
        # if smooth2 is lower than 1, the use smooth2=1; otherwise, round it and transform it to integer
        if smooth2 <1:
            smooth2=1
        else: 
            smooth2=int(np.round(smooth2))
        #Apply gaussian filter to dX and dY
        dX=pd.Series(dX).rolling(2*smooth2, center=True, win_type='gaussian').mean(std=smooth2).round(3)
        dY=pd.Series(dY).rolling(2*smooth2, center=True, win_type='gaussian').mean(std=smooth2).round(3)
        dX=dX.values
        dY=dY.values
    #Use Pitagoras to calculate the differential position (cat2+cat2=hyp2)
    dPos=np.sqrt(dX**2+dY**2) 
    #Calculate speed, dividing dPos by dT
    speed=dPos/dT
    return(speed)

def position_of_event(X_values, Y_values, timestamps, events, function=np.median, plot=True):
    """
    Calculate the average position at with an event occurs and the distance to that event at each timestamp

    Parameters
    ----------
    X_values : np.ndarray of shape (1 X n_timestamps)
        X_centroid positions at each timestamp. Must be of the same lenght than Y_values and timestamps
    Y_values : np.ndarray of shape (1 X n_timestamps)
        Y_centroid positions at each timestamp. Must be of the same lenght than X_values and timestamps
    timestamps : np.ndarray of shape (1 X n_timestamps)
        Timestamps. Must be of the same lenght than X_values and Y_values.
    events : one dimensional numpy.array or list of floats
        Timestamps of the event.
    function : function, optional
        Function to be used to estimate the most likely position of the event. The default is np.median.
    plot : bool, optional
        If True, wiil plot all the trajectory, position of each event and the most likely position of the event. The default is True.

    Raises
    ------
    TypeError
        If X_values, Y_values or timestamps is not np.ndarray.
    error
        DESCRIPTION.
    ValueError
        If X_values lenght, Y_values lenght and timestamps lenght are not the same.

    Returns
    -------
    Event most likely position in X dimension (float) and in in Y dimension (float), and distance to event (np.ndarray of shape (1 x n_timestamps)).

    """
    # if type(events)==dict:
    #     events_df=pd.DataFrame({key:pd.Series(value) for key,value in events.items()}).melt(var_name='Event', value_name='Timestamp').dropna().set_index('Timestamp').sort_index()
    #     events_df.index=np.round(event_df.index, 3)
    # elif type(events)==pd.DataFrame:
    #     events_df=events.dropna()
    # else:
    #     raise TypeError ('"events" must be either a dictionary with keys="Event names":values=list of timestamps, or a pandas.DataFrame with "Event" column corresponding to event name, and timestamps as index')
     #Evaluate if X_values, Y_values and timestamps ahve the same lenght, raise error if not
    if len(X_values)!=len(Y_values) or len(X_values)!=len(timestamps):
        raise ValueError ('X_values, Y_values and timestamps must have the same lenght')
     #Evaluate if X_values, Y_values and timestamps are of type np.array, raise error if not
    if type(X_values)!=np.ndarray or type(Y_values)!=np.ndarray or type(timestamps)!=np.ndarray:
        raise TypeError ('X_values, Y_values and timestamps must be of type numpy.ndarray')
    
    
    df=pd.DataFrame(columns=['X','Y'], index=timestamps, data=np.array([X_values,Y_values]).T)
    event_df=pd.DataFrame(columns=['Event'],index=events, data='event')
    df=incorporate_events_to_continuous(df,event_df,exact=False)
    event_XPos=function(df.loc[df['Event']=='event','X'])
    event_YPos=function(df.loc[df['Event']=='event','Y'])
    
    #plot
    if plot==True or type(plot)==str:
        plt.plot(df['X'],df['Y'],color='gray',alpha=0.5, label='Trajectory')
        plt.scatter(df.loc[df['Event']=='event','X'],df.loc[df['Event']=='event','Y'], color='black',s=10, label='Event positions')
        plt.scatter(event_XPos,event_YPos, color='red', marker='D',s=50, label='Event expected position')
        plt.legend(framealpha=0.5)
        plt.axis("equal")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        if type(plot)==str:
            plt.title(plot+' positions')
        else:
            plt.title('Event positions')
        plt.show()
    
    dist=np.sqrt((X_values-event_XPos)**2+(Y_values-event_YPos)**2)
    return(event_XPos,event_YPos,dist)
    # XPos_low_ci=np.quantile(df.loc[df['Event']=='event','X'],0.025)
    # XPos_high_ci=np.quantile(df.loc[df['Event']=='event','X'],0.975)
    
def clasify_events_base_on_time(event1,event2,treshold,mode='left'):
    """
    Clasiffy an event in two categories based on how close in time it occurs from an event of reference.

    Parameters
    ----------
    event1 : numpy.array of shape (1 x n)
        Event to classify.
    event2 : numpy.array of shape (1 x m)
        Event of reference.
    treshold : TYPE
        Threshold amount of time used to classify events.
    mode : str, optional
        Define the mode of evaluation of proximity. "left", only looks event1 that occur before event2; "right", 
        only looks event1 that ocurr after event2; "both", look temporal proximity before and after. The default is 'left'.

    Returns
    -------
    near : np.array of shape (1 x o)
        Subset of event1 classified as temporally close to event2.
    far : np.array of shape (1 x p).
        Subset of event1 classified as temporally far from event2.

    """
    near=[]
    far=[]
    for i in event1:
        j=event2-i
        if mode=='left':
            if len(j[j>0])>0 and np.min(j[j>0])<=treshold:
            
        # if np.min(abs(event2-i))<=treshold:
                near.append(i)
            else:
                far.append(i)
        elif mode == 'two-sides':
            if np.min(abs(j))<=treshold:
                near.append(i)
            else:
                far.append(i)
        else:
            if len(j[j<0])>0 and abs(np.max(j[j<0]))<=treshold:
                near.append(i)
            else:
                far.append(i)
    near=np.array(near)
    far=np.array(far)
    return near, far


def perievent_histogram(continuous, events, lower_limit, higher_limit, number_of_decimals=3):
    """
    Plot peri-event histograms and return dataframe with values

    Parameters
    ----------
    continuous : pandas.DataFrame
        Dataframe with continuous variables to be plotted in each column, and timestamps as index.
    events : list, dict, pandas.DataFrame
        Timestamps of each event.If a list, values must be the timestamps of the event. If more than one type of event is analyzed,
        a dict or pandas.DataFrame type must be used. If a dict, keys must be events name as str and values must be timestamps of 
        each event as list, numpy.array or pandas.Serie; if a pandas.DataFrame, it must be a column named "Event" with event's names, 
        and timestamps as index.
    lower_limit : float
        Lower limit of the time interval.
    higher_limit : float
        Higher limit of the time interval.
    number_of_decimals : int, optional
        Number of decimals of the peri-event time; it should be the same as the number of decimals in the index of continuous paramater.
        The default is 3.

    Raises
    ------
    TypeError
        If continuous type is not pandas.DataFrame or if events type is neither list, dict or pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame with the numeric results of the peri-event histograms.

    """
    if type(events)!=dict and type(events)!=pd.DataFrame and type(events)!=list:
        raise TypeError ('"events" must be either a list of timestamps, a dictionary with keys="Event names":values=list of timestamps, or a pandas.DataFrame with "Event" column corresponding to event name, and timestamps as index')
    if type(events)==list:
        events={'event':events}
    df=incorporate_events_to_continuous(continuous,events, exact=False)
    df=peri_event_time(continuous, pd.DataFrame(df['Event']), lower_limit, higher_limit, number_of_decimals, all_trials=False)
    with sns.plotting_context("poster"):
        for i in continuous.columns:
            g=sns.relplot(x='Peri-event_time', y=i, data=df,kind='line', hue='Event')
            g.map(plt.axvline,x=0,ls="--",c="gray")
            plt.title(i)
            plt.show()
    return df


def incorporate_events_to_continuous(continuous, events, exact=True):
    if type(events)==dict:
        event_df=pd.DataFrame({key:pd.Series(value) for key,value in events.items()}).melt(var_name='Event', value_name='Timestamp').dropna().set_index('Timestamp').sort_index()
        event_df.index=np.round(event_df.index, 3)
    elif type(events)==pd.DataFrame:
        event_df=events.copy()
    else:
        raise TypeError ('"events" must be either a dictionary with keys="Event names":values=list of timestamps, or a pandas.DataFrame with "Event" column corresponding to event name, and timestamps as index')
    if type(continuous) != pd.DataFrame:
        raise TypeError ('"continuous" must be a pandas.DataFrame with continuous values as columns and timestamps as index')

    if exact==True:       
        result=continuous.merge(event_df, how='left',left_index=True, right_index=True, validate='one_to_one')
    else:
        #Add events to spectogram dataframe
        continuous2=continuous.reset_index()
        continuous2.rename(columns={'index':'Time'},inplace=True)
        # event_df.reset_index(inplace=True)

        # reverse_merge=pd.merge_asof(event_df,continuous2.reset_index(), on='Time', direction='nearest')
        reverse_merge=pd.merge_asof(event_df,continuous2.reset_index(), right_on='Time', left_index=True,direction='nearest')
        result=continuous2.merge(reverse_merge[['Event','index']].set_index('index'),how='left', left_index=True, right_index=True)
        result.set_index('Time', inplace=True)
    return result