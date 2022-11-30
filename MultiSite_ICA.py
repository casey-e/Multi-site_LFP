# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:32:11 2022

@author: casey.e
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
# import pingouin as pg
from scipy import signal
import sys
import statsmodels.formula.api as smf
import time
# from mne_connectivity import spectral_connectivity_epochs

sys.path.append('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Scripts/Python scripts')
import nexfile
reader = nexfile.Reader(useNumpy=True)
from sklearn.decomposition import FastICA, PCA



region_channels_dict_molex={'vHip':['TETFP01','TETFP02','TETFP03','TETFP04'],
                            'BLA':['TETFP05','TETFP06','TETFP07','TETFP08'], 
                            'NAc':['TETFP09','TETFP10','TETFP11','TETFP12'], 
                            'PFC':['TETFP13','TETFP14','TETFP15','TETFP16']}

region_channels_dict_cohort2={'NAc':['TETFP01','TETFP02','TETFP03','TETFP04'],
                              'PFC':['TETFP05','TETFP06','TETFP07','TETFP08'], 
                              'vHip':['TETFP09','TETFP10','TETFP11','TETFP12'], 
                              'BLA':['TETFP13','TETFP14','TETFP15','TETFP16']}
region_channels_dict_E_A1={'PFC':['TETFP01','TETFP02','TETFP03','TETFP04'],
                           'BLA':['TETFP05','TETFP06','TETFP07','TETFP08'], 
                           'NAc':['TETFP09','TETFP10','TETFP11','TETFP12'], 
                           'vHip':['TETFP13','TETFP14','TETFP15','TETFP16']}

region_channels_dict_AllOtherMice={'NAc':['TETFP01','TETFP02','TETFP03','TETFP04'],
                                   'BLA':['TETFP05','TETFP06','TETFP07','TETFP08'], 
                                   'PFC':['TETFP09','TETFP10','TETFP11','TETFP12'], 
                                   'vHip':['TETFP13','TETFP14','TETFP15','TETFP16']}

mice_list=['Mouse1','Mouse2' ,'Mouse3','array1','array3','array7','array8','array5', 'Mouse4','Mouse5','Mouse6','Mouse7']
region_list=['vHip', 'PFC','BLA','NAc']




def timer (func):
    """ A decorator that prints how long does a functions took t run"""
    #Define wrapper function
    def wrapper(*args, **kwargs):
        #Set current time tostart_time
        start_time=time.time()
        #Call function and store the result in result
        result=func(*args, **kwargs)
        #Get total time the function took to run
        total_time=time.time()-start_time
        #Print the time the function took to run
        print('{} took {} s'.format(func.__name__, total_time))
        #Return the function as result
        return result
    return wrapper

def choose_region_chanels_dict():
    if mouse in ['Mouse1','Mouse2','Mouse3', 'Mouse6','Mouse7']:
        region_channels_dict=region_channels_dict_cohort2
    elif mouse in ['Mouse4', 'Mouse5', 'Mouse8']:
        region_channels_dict=region_channels_dict_molex
    elif mouse=='array1':
        region_channels_dict=region_channels_dict_E_A1
    else:
        region_channels_dict=region_channels_dict_AllOtherMice
    return region_channels_dict

@timer
def pull_variables_from_nex (file, event_of_interest=[], continuous_of_interest=[]):
    """
    Extract

    Parameters
    ----------
    file : TYPE string, optional
        DESCRIPTION. Name of the file to pull data from. Must have a nex5 extension.
    
    
    event_of_interest : TYPE list, optional
        DESCRIPTION. List of names of the events topull from the nex file. The default is empty list.

    Returns
    -------
    Dataframe with field potentials (in mV) in each column and timestaps as index.
    
    Dictionary with event names as keys and list of timestamps as values.
    
    Sampling rate of local field potentials (float).

    """
    fileData = reader.ReadNex5File(file)
    continuous={}
    continuous_list=[]
    events={}
    FP={}
    FP_list=[]

    for variable in fileData['Variables']:
        if 'FP' in variable['Header']['Name']:
            FP_list.append(variable)
            FP.update({variable['Header']['Name']:variable['ContinuousValues']})
        else:
            for event in event_of_interest:
                if event == variable['Header']['Name']:
                    events.update({event:variable['Timestamps']})
            for cont in continuous_of_interest:
                if cont == variable['Header']['Name']:
                    continuous_list.append(variable)
                    continuous.update({variable['Header']['Name']:variable['ContinuousValues']})
    
    
    #make timestamps    
    time_stamps=[]
    for i in range(len(FP_list[0]['FragmentIndexes'])): #iterate thorugh fragments
        first_ts=FP_list[0]['FragmentTimestamps'][i] # get the value of the first timestamp of the fragment
        ts=[(first_ts+x*FP_list[0]['Header']['SamplingRate']**(-1)) for x in range (FP_list[0]['FragmentCounts'][i])]
        time_stamps=time_stamps+ts # concatenate list of timestamps
    assert(len(time_stamps)==len(FP_list[0]['ContinuousValues'])) # evaluate that the number of timestaps is the same than the number of continuous values
    sampling_rate=FP_list[0]['Header']['SamplingRate']
    
    #Make dataframe with signal from every channel (LFP in mV) and timestamps as index
    df=pd.DataFrame(FP, index=time_stamps)
    # df.index=time_stamps #Put the timestamps as index
    df.index=np.round(df.index, 3) #Round timestamps to 3 decimals
    
    if len(continuous_of_interest)>=1:
        cont_df=pd.DataFrame(continuous, index=continuous_list[0]['FragmentTimestamps'].round(3))#Make df of other cont vars
        return df, sampling_rate, events, cont_df
    else:
        return df, sampling_rate, events



def spectogram(dataframe):
    """
    

    Parameters
    ----------
    dataframe : TYPE
        DESCRIPTION.
    start : TYPE
        DESCRIPTION.
    end : TYPE
        DESCRIPTION.

    Returns
    -------
    result_df : TYPE
        DESCRIPTION.

    """
    
    spectogram_dict={}
    for fp in dataframe.columns:
        f, t, Sxx = signal.spectrogram(dataframe[fp], sampling_rate,scaling='spectrum',nperseg=500,noverlap=400,nfft=2000) #run spectograms: t=time, f=frequencies, Sxx=power_values
        spect_df=pd.DataFrame(data=Sxx, index=f,columns=t).T
        spect_df=spect_df.loc[:,0:150]
        spectogram_dict.update({fp:spect_df})
    
    df_list=[]
    for region, channel_list in region_channels_dict.items():
        region_df_list=[]
        for fp,data in spectogram_dict.items():
            if fp in channel_list:
                region_df_list.append(data)
        df_region=sum(region_df_list)/len(region_df_list)
        rename_list=[(region+'_'+str(name)) for name in df_region.columns]
        df_region.rename(columns=dict(zip(list(df_region.columns), rename_list)), inplace=True)
        df_list.append(df_region)
    result_df=pd.concat(df_list, axis=1)
    means=result_df.mean(axis=0)
    sd=result_df.std(axis=0)
    result_df=result_df.subtract(means,axis=1).divide(means, axis=1)
    return result_df


def get_noisy_timestamps(dataframe, fold_std=4, value=0, quorum=0.6):
    """
    Identify timestamps with artifacts

    Parameters
    ----------
    dataframe : PANDAS DATAFRAME
        A pandas dataframe with the local field potential channels (or any other continuous variable)
        in each column.
    fold_std : NUMERIC, optional
        Folds of the standar deviation for a signal to be cosidered artifac, only used if value = 0. The default is 4.
    value : NUMERIC, optional
        Treshold for a signal's value to be considered artfact. If not provided, fold_std is used. The default is 0.
    quorum : FLOAT, optional
        Proportion of channels (columns) whose signal must overcome the treshold for the timestamp (row) to be classified as with artifact. The default is 0.6.

    Returns
    -------
    noise : PANDAS SERIE
        Serie of booleans, True for timestamps with artifacts, False for timestamps without artifacts.

    """
    boolean_df=pd.DataFrame(index=dataframe.index)
    if value !=0:
        boolean_df=abs(dataframe)>value
    else:
        for col in dataframe.columns:
            boolean_df[col]=abs(dataframe[col])>fold_std*dataframe[col].std()
    boolean_df=boolean_df*1
    noise=boolean_df.mean(axis=1)>quorum
    return noise


def delete_artifacts(dataframe, fold_std=4, value=0, quorum=0.6, plot=None):
    """
    Remove timestamps with artifacts

    Parameters
    ----------
    dataframe : PANDAS DATAFRAME
        A pandas dataframe with the local field potential channels (or any other continuous variable)
        in each column.
    fold_std : NUMERIC, optional
        Folds of the standar deviation for a signal to be cosidered artifac, only used if value = 0. The default is 4.
    value : NUMERIC, optional
        Treshold for a signal's value to be considered artfact. If not provided, fold_std is used. The default is 0.
    quorum : FLOAT, optional
        Proportion of channels (columns) whose signal must overcome the treshold for the timestamp (row) to be classified as with artifact. The default is 0.6.
    plot : INTEGER or None, optional
       If integer, plot the channel corresponding to the position indicated by plot, before and after removing timestamps with artifacts. If None, no ploting is made. The default is None.

    Returns
    -------
    dataframe: PANDAS DATAFRAME
    Dataframe with timestamps with artifacts removed

    """
    def plotting (dataframe, plot, title):
        if plot != None:
            _=plt.plot(dataframe.index,dataframe.iloc[:,plot])
            plt.title(title)
            plt.show()
    plotting(dataframe, plot, 'Raw data, with artifacts')
    noisy_timestamps=get_noisy_timestamps(dataframe, fold_std, value, quorum)
    dataframe=dataframe[~noisy_timestamps]
    plotting(dataframe, plot, 'Raw data, artifacts removed')
    return(dataframe)

# def rd_slice(df):
#     rd=rng.random()
#     if rd > 0.5:
#         train_df=df.iloc[(round(40*len(df)/100)):,:]
#     else:
#         train_df=df.iloc[:round((60*len(df)/100)),:]
#     test_df=df.drop(train_df.index)
#     return train_df, test_df

def rd_slice(df, train_prop=0.6, plot=None):
    df2=df.reset_index()
    start=rng.integers(0,(len(df2)-train_prop*len(df2)))
    end=start+round(train_prop*len(df2))
    train_df=df2.iloc[start:end,:]
    test_df=df2.drop(train_df.index)
    train_df.set_index('index', inplace=True)
    test_df.set_index('index', inplace=True)  
    #Plot a specified channels for train and test df
    if plot !=None:
        _=plt.plot(train_df.index,train_df.iloc[:,plot], c='blue', label='Train')
        _=plt.plot(test_df.index,test_df.iloc[:,plot], c='red',label='Test')
        plt.title('Raw data, splitted for train and test')
        plt.legend()
        plt.show()
    return train_df, test_df

def center_perievent_df_to_baseline_of_each_mouse(dataframe, baseline_start, baseline_end):
    normalized_list=[]
    for mouse in dataframe['MouseId'].unique():
        df=dataframe[dataframe['MouseId']==mouse]
        for col in df.columns:
            if 'IC_' in col:
                mean=df.loc[(df['Peri-event_time']>baseline_start)&(df['Peri-event_time']<baseline_end), col].mean()
                # std=df.loc[(df['Peri-event_time']>-10)&(df['Peri-event_time']<-7), col].std()
                df[col]=(df[col]-mean)
        normalized_list.append(df)
    result_df=pd.concat(normalized_list, ignore_index=True)
    return result_df

def zScore_perievent_df_to_baseline_of_each_mouse(dataframe, baseline_start, baseline_end):
    normalized_list=[]
    for mouse in dataframe['MouseId'].unique():
        df=dataframe[dataframe['MouseId']==mouse]
        for col in df.columns:
            if 'IC_' in col:
                mean=df.loc[(df['Peri-event_time']>baseline_start)&(df['Peri-event_time']<baseline_end), col].mean()
                std=df.loc[(df['Peri-event_time']>baseline_start)&(df['Peri-event_time']<baseline_end), col].std()
                df[col]=(df[col]-mean)/std
        normalized_list.append(df)
    result_df=pd.concat(normalized_list, ignore_index=True)
    return result_df

def zScore_Cocainedf_to_baseline_of_each_mouse(dataframe, baseline_start, baseline_end):
    normalized_list=[]
    for mouse in dataframe['MouseId'].unique():
        df=dataframe[dataframe['MouseId']==mouse]
        for col in df.columns:
            if 'IC_' in col:
                mean=df.loc[(df['Time(min)']>baseline_start)&(df['Time(min)']<baseline_end), col].mean()
                std=df.loc[(df['Time(min)']>baseline_start)&(df['Time(min)']<baseline_end), col].std()
                df[col]=(df[col]-mean)/std
        normalized_list.append(df)
    result_df=pd.concat(normalized_list, ignore_index=True)
    return result_df

def make_sham_events(events_to_control, timestamps, events_to_avoid={}, lower_lim=0, higher_lim=0, avoid_begining=0, avoid_end=0):
    if type(events_to_control)!=dict or type(events_to_avoid)!=dict:
        raise TypeError('events must be dictionary')
    print('making timestamps_to_avoid list')
    timestamps_to_avoid=[]
    for val in events_to_avoid.values():
        timestamps_to_avoid=timestamps_to_avoid+(list(val))
    print('removing begining and end from timestamps')
    filtered_timestamps=np.array(timestamps)
    filtered_timestamps=filtered_timestamps[(filtered_timestamps>(filtered_timestamps[0]+avoid_begining))&(filtered_timestamps<(filtered_timestamps[-1]-avoid_end))]
    print('removing timestamps_to_avoid')
    for ts in timestamps_to_avoid:
        filtered_timestamps=filtered_timestamps[(filtered_timestamps<(ts-lower_lim))|(filtered_timestamps>(ts+higher_lim))]
    print('making vars_dict')
    vars_dict={}
    for key,value in events_to_control.items():
        print(key)
        name='sham_'+key
        values=np.random.choice(filtered_timestamps, size=len(value), replace=False)
        vars_dict.update({name:values})
    return(vars_dict)
        

def set_variable_from_list_and_filename(filename, name_list, default=None):
    result=default
    for name in name_list:
        if name in filename:
            result=name
    return result

def events_count_in_sliced_df(df, events_dict):
    amount_of_events=np.concatenate(list(events_dict.values()))
    amount_of_events=amount_of_events[(amount_of_events>=df.index[0]) & (amount_of_events<=df.index[-1])]
    return(len(amount_of_events))



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

        


def delete_time_gaps(df, sampling_rate):
#Get rid of pauses. Replace jumps in timestamps as it had never been pauses by creating a 'Time' veriable using the first timestamp and the sampling rate, and set it as index
    df2=df.copy()
    df2.reset_index(inplace=True)
    df2['Time']=df2.loc[0,'index']+(1/sampling_rate)*df2.index
    df2.drop(columns=['index'],inplace=True)
    df2.set_index('Time', drop=True, inplace=True)
    return(df2)



#Make peri-event_time column
def peri_event_time(continuous, events, lower_limit, higher_limit, number_of_decimals=3, all_trials=False):
    if type(events)==dict:
        events_df=pd.DataFrame({key:pd.Series(value) for key,value in events.items()}).melt(var_name='Event', value_name='Timestamp').dropna().set_index('Timestamp').sort_index()
        events_df.index=np.round(event_df.index, 3)
    elif type(events)==pd.DataFrame:
        events_df=events.dropna()
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
            
  # df3['Peri-event_time']=df3['Peri-event_time'].round(1)
  # #Add mouseId and Day columns
  # df3['MouseId']=mouse
  # #Add spectogram dataframe to dict
  # SelfStim_full_df_dict.update({mouse:df3})
  # print('Done '+mouse)
  
def scale_centroids(centroids, x_dim,y_dim):
    
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

## This function is wrong because it applies Gaussian filter on speed instead of on dX and dT, which cause short movements to impact the speed
## However, I left it here as it could be useful to analyze short movements (as might be motion in neuroexplorer)
# def calculate_speed(X_values, Y_values, timestamps, smooth=0):
#     #Calculate the differential X and Y, by substracting each value to the next value (using shift)
#     # dif_X=np.array(scaled_centroids['Centroid_X (cm)']-scaled_centroids['Centroid_X (cm)'].shift(periods=1))
#     # dif_Y=np.array(scaled_centroids['Centroid_Y (cm)']-scaled_centroids['Centroid_Y (cm)'].shift(periods=1))
    
#     dif_X=np.roll(X_values,-1)-X_values
#     dif_X[-1]=np.nan
#     dif_Y=np.roll(Y_values,-1)-Y_values
#     dif_Y[-1]=np.nan
#     #Use Pitagoras to calculate the differential position (cat2+cat2=hyp2)
#     dif_position=np.sqrt(dif_X**2+dif_Y**2)
#     #Calculate nstantaneus change in time
#     # dif_time=np.round(np.array((scaled_centroids.index-np.roll(scaled_centroids.index, 1))),3)
#     dif_time=np.round(np.roll(timestamps, -1)-timestamps,3)
#     dif_time[-1]=np.nan
#     #Calculate average delta time, which will be used for the smoothing process
#     avg_dif_time=np.mean(dif_time[0:-1])
#     #Calculate speed, dividing in
#     speed=dif_position/dif_time
#     #Calculate the number of rows to use a std in the gaussian filter, by dividing the smooth time provided by the average delta time
#     smooth2=smooth/avg_dif_time
#     print(dif_time)
#     print(avg_dif_time)
#     print(smooth2)
#     if smooth2!=0:  # Only apply smoothing if smooth is different from 0  
#         if smooth2 <1:
#             smooth2==1
#         else: 
#             smooth2=int(np.round(smooth2))
#         print(avg_dif_time)
#         print(smooth2)
#         speed=pd.Series(speed).rolling(2*smooth2, center=True, win_type='gaussian').mean(std=smooth2).round(3)
#         speed=speed.values
#     #align events to centroids df
#     return(speed,dif_position)

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
    speed: NUMPY ARRAY
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
    if plot==True:
        plt.plot(df['X'],df['Y'],color='gray',alpha=0.5, label='Trajectory')
        plt.scatter(df.loc[df['Event']=='event','X'],df.loc[df['Event']=='event','Y'], color='black',s=10, label='Event positions')
        plt.scatter(event_XPos,event_YPos, color='red', marker='D',s=50, label='Event expected position')
        plt.legend()
        plt.show()
    
    dist=np.sqrt((X_values-event_XPos)**2+(Y_values-event_YPos)**2)
    return(dist)
    # XPos_low_ci=np.quantile(df.loc[df['Event']=='event','X'],0.025)
    # XPos_high_ci=np.quantile(df.loc[df['Event']=='event','X'],0.975)
    
def clasify_events_base_on_time(event1,event2,treshold,mode='left'):
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

def make_intervals_based_on_continuous_signal(continuous, timestamps, treshold, minimum_duration):
    rest=np.where(continuous>=treshold,1,0) #Identify resting periods based using a treshold of speed=0.5 cm/seg
    labelled_rest,number_of_feaures=label(rest) # label every period of consecutive rest with a nuumber
    slices=find_objects(labelled_rest) # make slices for every period
    # long_slices=[i for i in slices if labelled_rest[i].size>150] #filter for periods longer than 5 seconds 
    start_index=[i[0].start for i in slices] #sample long_slices
    stop_index=[i[0].stop for i in slices] #sample long_slices
    rest_starts=np.array(timestamps[start_index])
    rest_stops=np.array(timestamps[stop_index])
    durations_criteria=rest_stops-rest_starts>=minimum_duration
    rest_starts2=rest_starts[durations_criteria]
    rest_stops2=rest_stops[durations_criteria]
    return rest_starts, rest_stops
    
## Create movement stop event and filter the events close to pellet removal
from scipy.ndimage import label, find_objects # import neccesary functions from scipy
rest=np.where(scaled_centroids['Speed'].values<0.5,1,0) #Identify resting periods based using a treshold of speed=0.5 cm/seg
labelled_rest,number_of_feaures=label(rest) # label every period of consecutive rest with a nuumber
slices=find_objects(labelled_rest) # make slices for every period
long_slices=[i for i in slices if labelled_rest[i].size>150] #filter for periods longer than 5 seconds 
# indexes=np.sort(np.random.choice(list(range(len(long_slices))), size=len(events['left_poke']), replace=False)) #Make indxes to sample long_slices, of the same lenght than left_poke
# slices_sampled=[long_slices[i] for i in indexes] #sample long_slices

start_index=[i[0].start for i in long_slices] #sample long_slices
stop_index=[i[0].stop for i in long_slices] #sample long_slices
# start_index, stop_index=[slices_sampled[i][0].start for i in slices_sampled]
rest_starts=np.array(scaled_centroids.reset_index().loc[start_index, 'index'])
rest_stops=np.array(scaled_centroids.reset_index().loc[stop_index, 'index'])
#Discard events closer than 10 seconds from pellet remova
contaminated_rest_starts, rest_starts=clasify_events_base_on_time(rest_starts, events['pellet_removal'],30,mode='two-sides')

events.update({'rest_starts':rest_starts, 'contaminated_rest_starts':contaminated_rest_starts})
    
    

#%%
### Set behaviors to use to train ica by including them in behaviors_lis. ###

behaviors_list=['infusions_cohort1','infusions_cohort2']



infusions_cohort2_dict={'directory':'C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Intraoral_infusions/Cohort2/pl2_files_renamed',
                'events_list':['Water', 'Sucrose_5%','Sucrose_15%'], 'days':['Day1','Day2']}

infusions_cohort1_dict={'directory':'C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Intraoral_infusions/Cohort 1/Pre_HFD/Processed_Nex_files',
                'events_list':['Water', 'Sucrose_5%','Sucrose_15%'], 'days':[]}

voluntary_feeding_dict={'directory':'C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Over night freely feeding with FED/Nex_files',
                'events_list':['pellet_removal'], 'days':[]}

SelfStim_dict={'directory':'C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/VTA_opto_stimulation/Self stim/Self stim over time with FEDs/Ephy&photometry/Nex_files',
                'events_list':['Self_stim'], 'days':[]}

voluntary_drinking_head_fixed_dict={'directory':'C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Voluntary_drinkingHeadFixedandLickometer/Curve of sucrose test/Test before HFD/Nex5_files/Processed_Nex_files',
                'events_list':['Water_drink','Sucrose1%_drink','Sucrose15%_drink'], 'days':[]}

behaviors_dict={'infusions_cohort2':infusions_cohort2_dict,'infusions_cohort1':infusions_cohort1_dict, 'voluntary_feeding':voluntary_feeding_dict,'SelfStim':SelfStim_dict, 'voluntary_drinking_head_fixed':voluntary_drinking_head_fixed_dict}


#%%
###     slice dataframe from each experiment, make spectograms and and concatenate. Make a train and a test dataframe. Train ica       ###

full_train_dict={}
full_test_dict={} # this dict will have dataframes for spectograms to train the ICA
included_events_dict={}
rng = np.random.default_rng(92)

for behavior in behaviors_list:
    current_behavior_dict=behaviors_dict[behavior]
    directory=current_behavior_dict['directory']
    events_list=current_behavior_dict['events_list']
    day_list=current_behavior_dict['days']
    os.chdir(directory)

    for file in glob('*.nex5'):
        mouse=set_variable_from_list_and_filename(file, mice_list)
        day=set_variable_from_list_and_filename(file, day_list, default='unique_day')
        region_channels_dict=choose_region_chanels_dict() #Function to choose the right dictionary according to the mouse
        df,events,sampling_rate=pull_variables_from_nex (file, events_list) #Read nex file and pull df of LFP, events and sampling rate
        # Remove noise and plot first channel to check removal of noise
        df=delete_artifacts(dataframe=df,value=0.9, quorum=0, plot=0)    
    
        # Make spectogram dataframe to train ica
        first_event=min([min(list(i)) for i in list(events.values())]) #Get first infusion
        last_event=max([max(list(i)) for i in list(events.values())]) # Get last infusion
        df=df.loc[(first_event-40):(last_event+40),:] #Slice dataframe from 40 sec before first event to 40 sec after last event
        train_df, test_df=rd_slice(df, plot=0) #Use rd_slice to make train and test dataframes, and plot to check splitting
        
        #Use spectogram function to make a dataframe with spectogram data from int_start to int_end: mouse_spectogram_df
        mouse_spectogram_df=spectogram(train_df)
        full_train_dict.update({mouse+'_'+day+behavior:mouse_spectogram_df}) # Put the spectogram dataframe in a dictionary with key=mouse_day, value=dataframe
        #Calculate how many events are in the sliced dataframe and add it to included_events_dict
        included_events_dict.update({behavior+mouse+day:events_count_in_sliced_df(train_df, events)})
        
        #Make test_df with the rest of the recording data
        mouse_spectogram_df=spectogram(test_df)#Make spectogram for test
        full_test_dict.update({mouse+'_'+day+behavior:mouse_spectogram_df})#Add to test dictionary

        print('Done '+file)
    print('Done '+behavior)

full_train_df=pd.concat(full_train_dict.values(), ignore_index=True) #Train dataframe with data from all files
full_test_df=pd.concat(full_test_dict.values(), ignore_index=True)  #Test dataframe with data from all files
print('Done making spectograms to train ICA!!!')
print('\nNumber of events included in training dataset, per file:\n')
for k,v in included_events_dict.items():
    print(k,':', v)
### Run this to save full_train and full_test dataframes
# os.chdir('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Independent Component Analysis')
# full_train_df.to_csv('full_train_spectogram_df_IntraoraInfusionsBothCohorts060222.csv')
# full_test_df.to_csv('full_test_spectogram_df_IntraoraInfusionsBothCohorts060222.csv')

#%%
### Calculate the best number of independent components ###
# full_train_df=pd.read_csv('full_train_spectogram_df_IntraoraInfusionsBothCohorts060222.csv', index_col=0)
# full_test_df=pd.read_csv('full_test_spectogram_df_IntraoraInfusionsBothCohorts060222.csv', index_col=0)

# Use a for loop to run ICA from 1 to 1150 cindependt components. For each one calculate explained variance and make a dataframe
#Select the max numberof components to try
max_n_components=300

df=[full_train_df,full_test_df]
n_array=np.empty((2,max_n_components))
explained_variance_array=np.empty((2,max_n_components))
aic_array=np.empty((2,max_n_components))

train_null_model_variance=full_train_df.values.var()
test_null_model_variance=full_test_df.values.var()
null_model_variances=[train_null_model_variance,test_null_model_variance]

n_array[:,0]=[0,0]
explained_variance_array[:,0]=[0.0,0.0]
aic_array[:,0]=[(len(full_train_df.index) * np.log(null_model_variances[0]))+(2*1),(len(full_test_df.index) * np.log(null_model_variances[1]))+(2*1)]
for n in range(1,max_n_components):
    print('Computing ICA with ',n,'sources ...')
    ica = FastICA(n_components=n, max_iter=400000)
    ic= pd.DataFrame(ica.fit_transform(full_train_df))  # Reconstruct signals
    mixing_matrix = pd.DataFrame(ica.mixing_)  # Get estimated mixing matrix
    weights=pd.DataFrame(ica.components_)


    for i in [0,1]:
#ic= ica.fit_transform(full_spectogram_df)  # Reconstruct signals
#mixing_matrix = ica.mixing_  # Get estimated mixing matrix
#weights=pd.DataFrame(ica.components_)
        #print('calculating explained variance...')
    #full_ICAspectogram_df_means=full_ICAspectogram_df.mean()
    #null_model_variance=((full_ICAspectogram_df.subtract(full_ICAspectogram_df_means, axis=1))**2).values.mean()
    #null_model_variance2=full_ICAspectogram_df.values.var()
        ic=pd.DataFrame(np.dot(df[i],weights.T))#Calculate ic for full_test_df
        reconstructed_df=pd.DataFrame(np.dot(ic,mixing_matrix.T))#Reconstruct full_test_df multiplying ic_test and mixing matrix
    
        ica_model_variance=((df[i].values-reconstructed_df.values)**2).mean()
        explained_variance=(null_model_variances[i]-ica_model_variance)/null_model_variances[i]
        aic=(len(df[i].index) * np.log(ica_model_variance))+(2*n)
    
        n_array[i,n]=n
        explained_variance_array[i,n]=explained_variance
        aic_array[i,n]=aic
    
    
   # full_train_df_recontructed=pd.DataFrame(ica.inverse_transform(ic))
    
    #ic_test=pd.DataFrame(np.dot(full_test_df,weights.T))#Calculate ic for full_test_df
    #full_test_df_recontructed=pd.DataFrame(np.dot(ic_test,mixing_matrix.T))#Reconstruct full_test_df multiplying ic_test and mixing matrix
    
    #ica_model_variance_train=((full_train_df.values-full_train_df_recontructed.values)**2).mean()
    #explained_variance_train=(train_null_model_variance-ica_model_variance_train)/train_null_model_variance
    
    #ica_model_variance_test=((full_test_df.values-full_test_df_recontructed.values)**2).mean()
    #explained_variance_test=(test_null_model_variance-ica_model_variance_test)/test_null_model_variance
    
    

ica_expl_var_train2=pd.DataFrame(np.array([n_array[0],explained_variance_array[0],aic_array[0]]).T)
ica_expl_var_train2.rename(columns={0:'#_IC',1:'Explained_variance',2:'Acaike'}, inplace=True)
ica_expl_var_train2['First_derivate']=ica_expl_var_train2['Explained_variance'].shift(-1)-ica_expl_var_train2['Explained_variance']
ica_expl_var_train2['Second_derivate']=ica_expl_var_train2['First_derivate'].shift(-1)-ica_expl_var_train2['First_derivate']
ica_expl_var_train2['Acaike_diff']=ica_expl_var_train2['Acaike']-ica_expl_var_train2['Acaike'].shift(1)
### Run this to save ica_expl_var_df
# os.chdir('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Independent Component Analysis')
# ica_expl_var_train2.to_csv('ica_expl_var_train_df_IntraoraInfusionsBothCohorts060222.csv')
# ica_expl_var_test2.to_csv('ica_expl_var_test_IntraoraInfusionsBothCohorts060222.csv')

### Run this to load ica_expl_var_df
#os.chdir('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Independent Component Analysis')
#ica_expl_var_df=pd.read_csv('ica_expl_var_df.csv')

ica_expl_var_test2=pd.DataFrame(np.array([n_array[1],explained_variance_array[1],aic_array[1]]).T)
ica_expl_var_test2.rename(columns={0:'#_IC',1:'Explained_variance',2:'Acaike'}, inplace=True)
ica_expl_var_test2['First_derivate']=ica_expl_var_test2['Explained_variance'].shift(-1)-ica_expl_var_test2['Explained_variance']
ica_expl_var_test2['Second_derivate']=ica_expl_var_test2['First_derivate'].shift(-1)-ica_expl_var_test2['First_derivate']
ica_expl_var_test2['Acaike_diff']=ica_expl_var_test2['Acaike']-ica_expl_var_test2['Acaike'].shift(1)


#The previously calculated Acaike is wrong, as it uses number of components as number of paramaters, but the
# number of parameters actually is equal to number of components * number of features
# Because y would be features values and x would be the components; each sample is each timestamp
# so since to make the features matrix with the components matrix you have to multiply the components matrix 
# by the mixing matrix, the number of parameters will be the ammount of cells in the mixing matrix, wich is 
# #components * #featues (therefore, the #parameters willl be independent of the number of timestamps in the sample)
ica_expl_var_train2['Real_Acaike']=(len(df[0].index)*np.log(null_model_variances[0]-(ica_expl_var_train2['Explained_variance']*null_model_variances[0])))+(2*ica_expl_var_train2['#_IC']*1204)
ica_expl_var_train2['Real_Acaike_diff']=ica_expl_var_train2['Real_Acaike']-ica_expl_var_train2['Real_Acaike'].shift(1)

ica_expl_var_test2['Real_Acaike']=(len(df[1].index)*np.log(null_model_variances[1]-(ica_expl_var_test2['Explained_variance']*null_model_variances[1])))+(2*ica_expl_var_test2['#_IC']*1204)
ica_expl_var_test2['Real_Acaike_diff']=ica_expl_var_test2['Real_Acaike']-ica_expl_var_test2['Real_Acaike'].shift(1)

#ica_expl_var_df3=ica_expl_var_df2.drop(index=[0,1,2])
for name,df in zip(['train','test'],[ica_expl_var_train2, ica_expl_var_test2]):
    #Plot expl variance vs #IC
    
    # ax=plt.plot(df['#_IC'],df['Explained_variance'], marker='o')
    # plt.xlabel('#_IC', fontsize=30)
    # plt.ylabel('Explained_variance', fontsize=30)
    # # plt.title(name+'_Explained variance')
    # # plt.xlim(-1,50)
    # #plt.ylim(0.62,0.75)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # # for axis in ['bottom','left']:
    # #     ax.spines[axis].set_linewidth(4)
    # ax.spines['right']. set_visible(False)
    # ax.spines['top']. set_visible(False)
    # plt.show()
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='#_IC', y='Explained_variance', data=df)
        sns.despine(top=True,right=True)
        plt.xlabel('# IC', fontsize=30)
        plt.ylabel('Explained variance', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        # plt.xticks([-20,-10,0,10,20], fontsize=30)
        # plt.subplots_adjust(bottom=0.3)
        # plt.savefig('VoluntaryFeeding_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()

    #Plot first derivative vs #IC
    # _=plt.plot(df['#_IC'],df['First_derivate'], marker='o')
    # plt.xlabel('#_IC')
    # plt.ylabel('First_derivative')
    # plt.title(name+'_First derivativee')
    # plt.xlim(-1,50)
    # plt.show()
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='#_IC', y='First_derivate', data=df)
        sns.despine(top=True,right=True)
        plt.xlabel('# IC', fontsize=30)
        plt.ylabel('First derivative', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(0,50)
        # plt.xticks([-20,-10,0,10,20], fontsize=30)
        # plt.subplots_adjust(bottom=0.3)
        # plt.savefig('VoluntaryFeeding_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()
    

    #Plot second derivative vs #IC
    # _=plt.plot(df['#_IC'],df['Second_derivate'], marker='o')
    # plt.xlabel('#_IC')
    # plt.ylabel('Second_derivativee')
    # plt.title(name+'_Second derivativee')#plt.xlim(0,50)
    # plt.xlim(-1,50)
    # plt.hlines(y=0, xmin=-1,xmax=50, linestyles='--')
    # plt.show()
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='#_IC', y='Second_derivate', data=df)
        sns.despine(top=True,right=True)
        plt.hlines(y=0,xmin=0,xmax=50,linestyles='--',color='black')
        plt.xlabel('# IC', fontsize=30)
        plt.ylabel('Second derivative', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(0,50)
        # plt.xticks([-20,-10,0,10,20], fontsize=30)
        # plt.subplots_adjust(bottom=0.3)
        # plt.savefig('VoluntaryFeeding_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()

    #Plot Acaike and change in Acaike in the same graph
    # fig,ax1=plt.subplots()
    # ax1.plot(df['#_IC'],df['Real_Acaike'], marker='o')
    # plt.xlabel('#_IC')
    # plt.ylabel('Real_Acaike', color='blue')
    # plt.title(name+'_Real_Acaike')
    # #plt.xlim(-1,40)
    # #plt.ylim(0,100000)
    # #plt.hlines(y=0, xmin=0,xmax=40, linestyles='--',color='blue')
    # ax2=ax1.twinx()
    # ax2.plot(df['#_IC'],df['Real_Acaike_diff'], marker='o', color='orange')
    # #plt.ylim(-100,100)
    # plt.ylabel('Real_Acaike_diff', color='orange')
    # plt.hlines(y=0, xmin=0,xmax=45, linestyles='--',color='orange')
    # plt.show()
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='#_IC', y='Real_Acaike', data=df)
        sns.despine(top=True,right=True)
        plt.vlines(x=10,ymin=0,ymax=60000,linestyles='--',color='black')
        plt.xlabel('# IC', fontsize=30)
        plt.ylabel('Acaike', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(0,50)
        plt.ylim(15000,60000)
        # plt.xticks([-20,-10,0,10,20], fontsize=30)
        # plt.subplots_adjust(bottom=0.3)
        # plt.savefig('VoluntaryFeeding_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()
    
    print('Best number of IC for ',name,':',int(df.loc[df['Real_Acaike']==df['Real_Acaike'].min(), '#_IC'].values[0]))

#Choose the #IC that yields the minimum Acaike
best_number_of_IC=int(ica_expl_var_train2.loc[ica_expl_var_train2['Real_Acaike']==ica_expl_var_train2['Real_Acaike'].min(), '#_IC'].values[0])

#%%
## Run ICA with the optimal number of IC
print('Computing ICA with optimal number of IC='+str(best_number_of_IC)+'...')
ica = FastICA(n_components=best_number_of_IC, max_iter=10000000)
ic= pd.DataFrame(ica.fit_transform(full_train_df))  # Reconstruct signals
mixing_matrix = pd.DataFrame(ica.mixing_)  # Get estimated mixing matrix
weights=pd.DataFrame(ica.components_)
print('Done ICA!')

#assert np.allclose(full_ICAspectogram_df, np.dot(ic, mixing_matrix.T) + ica.mean_)

## Plot the components weights
weights.rename(columns=dict(zip(weights.columns, full_train_df.columns)), inplace=True)
os.chdir('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/OptoDBS2022/Poster/Intraoral_infusions/Weights')
for i in weights.index:
    weights_i=weights.loc[i,:]
    weights_i=pd.DataFrame(weights_i)
    weights_i.reset_index(inplace=True)
    weights_i.rename(columns={'index':'Frequency'}, inplace=True)
    for region_freq in weights_i['Frequency'].unique():
        for region in region_list:
            if region in region_freq:
                weights_i.loc[weights_i['Frequency']==region_freq, 'Region']=region
                weights_i.replace(region_freq, region_freq.replace(region,'').replace('_',''), inplace=True)
            #components=components.pivot(index='Frequency',columns='Region',values='weight')
            #components.reset_index(drop=True,inplace=True)
    weights_i['Frequency']=weights_i['Frequency'].astype(float)
    weights_i.sort_values(by='Frequency', inplace=True)
    g=sns.heatmap(weights_i.pivot(index='Region',columns='Frequency',values=i), robust=True, center=0, cbar=False)
    g.set_ylabel('Region', fontsize=20)
    g.set_xlabel('Freq. (Hz)', fontsize=20)
    plt.yticks(fontsize=28)
    plt.xticks([0,60,120,180,240,300],[0,30,60,90,120,150], fontsize=28,rotation=0)
    plt.tick_params(labelsize=15) 
    # g.collections[0].colorbar.ax.tick_params(labelsize=13)
    plt.title('Functional network '+str(i+1), fontsize=22)
    plt.show()
    plt.subplots_adjust(bottom=0.25, left=0.2)
    # plt.savefig('Source_'+str(i+1)+'.pdf')

# Find main features of each component by transforming the weights in 1s if the value>2std and 0s if value<2std for each component
weights_std=weights.std(axis=1)
weights_binary=pd.DataFrame(index=list(range(10)), columns=weights.columns)
for i in weights.index:
    weights_binary.loc[i,:]=abs(weights.loc[i,:])>(1.6*weights_std[i])
weights_binary=weights_binary*1
weights_filtered=weights*weights_binary
for i in weights_filtered.index:
    weights_i=weights_filtered.loc[i,:]
    weights_i=pd.DataFrame(weights_i)
    weights_i.reset_index(inplace=True)
    weights_i.rename(columns={'index':'Frequency'}, inplace=True)
    for region_freq in weights_i['Frequency'].unique():
        for region in region_list:
            if region in region_freq:
                weights_i.loc[weights_i['Frequency']==region_freq, 'Region']=region
                weights_i.replace(region_freq, region_freq.replace(region,'').replace('_',''), inplace=True)
            #components=components.pivot(index='Frequency',columns='Region',values='weight')
            #components.reset_index(drop=True,inplace=True)
    weights_i['Frequency']=weights_i['Frequency'].astype(float)
    weights_i.sort_values(by='Frequency', inplace=True)
    g=sns.heatmap(weights_i.pivot(index='Region',columns='Frequency',values=i), center=0, robust=True)
    plt.title('Source '+str(i+1))
    plt.show()


#%%

#Project ica to voluntary whole feeding recordings and use components to predict consumption
directory='C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/Over night freely feeding with FED/Nex_files'
os.chdir(directory)

vol_feeding_df_list=[] # this list will have dataframes for independent components of each whole recording
event_of_interest=['left_poke', 'pellet_removal', 'right_poke']
centroids=['Centroid1_X', 'Centroid1_Y']

for file in zz:
    mouse=set_variable_from_list_and_filename(file, mice_list)
    # day=set_variable_from_list_and_filename(file, day_list, default='unique_day')
    region_channels_dict=choose_region_chanels_dict() #Function to choose the right dictionary according to the mouse
    df,sampling_rate, events,centroids_df=pull_variables_from_nex ( file, event_of_interest,centroids) #Read nex file and pull df of LFP, events and sampling rate
     
    #Create sham_left_poke using the make_sham_events function
    # events.update(make_sham_events(events, df.index, events, 8,8,12,12))
    
    # #Create mistake_right_poke and exploratory_right_poke
    # mistake_right_poke, exploratory_right_poke=clasify_events_base_on_time(events['right_poke'], events['left_poke'],30)
    # events.update({'mistake_right_poke':mistake_right_poke, 'exploratory_right_poke':exploratory_right_poke})
    # del(events['right_poke'])
    # Remove noise and plot first channel to check removal of noise
    df=delete_artifacts(dataframe=df,value=0.9, quorum=0, plot=0)
    
    
    ## Identify movement initiations, align pellet retrieval to position
    #Remove zeros from centroids_df
    centroids_df.replace(0,np.nan, inplace=True)
    centroids_df.dropna(inplace=True)
    centroids_df=centroids_df.loc[(centroids_df['Centroid1_X']<=np.quantile(centroids_df['Centroid1_X'], 0.999))&(centroids_df['Centroid1_Y']<=np.quantile(centroids_df['Centroid1_Y'], 0.999))]
    #Scale centroids_df
    scaled_centroids=pd.DataFrame(scale_centroids(centroids_df.values.T,17,30).T, index=centroids_df.index, columns=['Centroid_X (cm)','Centroid_Y (cm)'])
    plt.plot(scaled_centroids['Centroid_X (cm)'],scaled_centroids['Centroid_Y (cm)'])
    plt.show()
    
    
    ##Calculate speed
    scaled_centroids['Speed']=calculate_speed(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),1)
    
    ###### Consider using the peak of speed to identify movement initialization as a simpler solution: use scipy findpeaks
    ## Create movement initiation event
    #Use find peaks from scipy to find peaks in speed above 2 cm/sec and that that last at least 1.5 sec
    movement_initiation, properties= signal.find_peaks(scaled_centroids['Speed'], height=2, width=45) #Get peak's rows
    movement_initiation=np.array(scaled_centroids.reset_index().loc[movement_initiation,'index']) # Get timestamps corresponding to peak's rows
    del(properties) # Properties won't be used
    movement_initiation=np.sort(np.random.choice(movement_initiation, size=len(events['left_poke']), replace=False)) #Take a random sample of the same lenght than left_poke event
    events.update({'movement_initiation':movement_initiation})
    del(movement_initiation)
    
    # ## Create movement stop event
    # from scipy.ndimage import label, find_objects # import neccesary functions from scipy
    # rest=np.where(scaled_centroids['Speed'].values<0.5,1,0) #Identify resting periods based using a treshold of speed=0.5 cm/seg
    # labelled_rest,number_of_feaures=label(rest) # label every period of consecutive rest with a nuumber
    # slices=find_objects(labelled_rest) # make slices for every period
    # long_slices=[i for i in slices if labelled_rest[i].size>150] #filter for periods longer than 5 seconds
    # #Discard events closer than 10 seconds from pellet removal    
    # indexes=np.sort(np.random.choice(list(range(len(long_slices))), size=len(events['left_poke']), replace=False)) #Make indxes to sample long_slices, of the same lenght than left_poke
    # # slices_sampled=[long_slices[i] for i in indexes] #sample long_slices
    # start_index=[long_slices[i][0].start for i in indexes] #sample long_slices
    # stop_index=[long_slices[i][0].stop for i in indexes] #sample long_slices
    # # start_index, stop_index=[slices_sampled[i][0].start for i in slices_sampled]
    # rest_starts=np.array(scaled_centroids.reset_index().loc[start_index, 'index'])
    # rest_stops=np.array(scaled_centroids.reset_index().loc[stop_index, 'index'])
    # events.update({'rest_starts':rest_starts, 'rest_stops':rest_stops})
    
    ## Create movement stop event and filter the events close to pellet removal
    from scipy.ndimage import label, find_objects # import neccesary functions from scipy
    rest=np.where(scaled_centroids['Speed'].values<0.5,1,0) #Identify resting periods based using a treshold of speed=0.5 cm/seg
    labelled_rest,number_of_feaures=label(rest) # label every period of consecutive rest with a nuumber
    slices=find_objects(labelled_rest) # make slices for every period
    long_slices=[i for i in slices if labelled_rest[i].size>450] #filter for periods longer than 5 seconds 
    # indexes=np.sort(np.random.choice(list(range(len(long_slices))), size=len(events['left_poke']), replace=False)) #Make indxes to sample long_slices, of the same lenght than left_poke
    # slices_sampled=[long_slices[i] for i in indexes] #sample long_slices
   
    start_index=[i[0].start for i in long_slices] #sample long_slices
    stop_index=[i[0].stop for i in long_slices] #sample long_slices
    # start_index, stop_index=[slices_sampled[i][0].start for i in slices_sampled]
    rest_starts=np.array(scaled_centroids.reset_index().loc[start_index, 'index'])
    rest_stops=np.array(scaled_centroids.reset_index().loc[stop_index, 'index'])
    #Discard events closer than 10 seconds from pellet remova
    contaminated_rest_starts, rest_starts=clasify_events_base_on_time(rest_starts, events['pellet_removal'],30,mode='two-sides')
    
    events.update({'rest_starts':rest_starts, 'contaminated_rest_starts':contaminated_rest_starts})
    
    rest_starts2, rest_stops2 = make_intervals_based_on_continuous_signal(scaled_centroids['Speed'].values, scaled_centroids.index, 0.5, 15)
    # contaminated_rest_starts2, rest_starts2=clasify_events_base_on_time(rest_starts2, events['pellet_removal'],30,mode='two-sides')
    
    # # scaled_centroids_subset=scaled_centroids.loc[30070:30120,:].reset_index()
    # # movement_initiation, properties= signal.find_peaks(scaled_centroids_subset['Speed'], height=2, width=30)
    # # #Plot centroids and speed vs timme
    # plt.plot(scaled_centroids_subset.index,scaled_centroids_subset['Centroid_X (cm)'], label='X position')
    # plt.plot(scaled_centroids_subset.index,scaled_centroids_subset['Centroid_Y (cm)'], label='Y position')
    # plt.plot(scaled_centroids_subset.index,scaled_centroids_subset['Speed'], label='Speed')
    # plt.plot(movement_initiation,scaled_centroids_subset.loc[movement_initiation,'Speed'], 'o',color='black')
    # # plt.hlines(y=2,xmin=30070,xmax=30120)
    # plt.legend()
    # plt.xlabel('Time(sec)')
    # plt.show()
    #Calculate distance to the position of left poke. Use events['left_poke'][(events['left_poke']>np.min(scaled_centroids.index))&(events['left_poke']<np.max(scaled_centroids.index))]
    # as the event parameter, to avoid using events that happened when the camera wasn't working (lights off)
    
    scaled_centroids['Distance_to_left_poke']=position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events['left_poke'][(events['left_poke']>np.min(scaled_centroids.index))&(events['left_poke']<np.max(scaled_centroids.index))], np.median)
    
    #Run this with different events to verify the function works
    for event in events.keys():
        position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events[event])
        print(event)
    
    
    #Incorporate centroids to df
    df2=incorporate_events_to_continuous(scaled_centroids,events, exact=False)
    # df2=df2.loc[0:42767.9,]
    df3=df2.copy()
    # df3[['Centroid_X (cm)', 'Centroid_Y (cm)', 'Speed', 'Distance_to_left_poke']].fillna(method='ffill', inplace=True)
    df3=peri_event_time(df2.drop(columns='Event'), pd.DataFrame(df2['Event']), -20,20,2,all_trials=False)
    df3=df3[(df3['Event']!='exploratory_right_poke')&(df3['Event']!='mistake_right_poke')]
    with sns.plotting_context("poster"):
        g=sns.relplot(x='Peri-event_time', y='Speed', data=df3,kind='line', hue='Event')
        # g.fig.suptitle('Functional network '+i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=0.5,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
    #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
        g.set_ylabels('Speed (cm/sec)')
        g.set_xlabels('Time (sec)')
    #plt.ylim(-0.002,0.002)
        plt.show()
        
    with sns.plotting_context("poster"):
        g=sns.relplot(x='Peri-event_time', y='Distance_to_left_poke', data=df3,kind='line', hue='Event')
        # g.fig.suptitle('Functional network '+i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=1,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
    #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
        g.set_ylabels('Distance to left poke (cm)')
        g.set_xlabels('Time (sec)')
    #plt.ylim(-0.002,0.002)
        plt.title('rest_starts not filtered')
        plt.show()
    
    
    
    df2.loc[(df2['Distance_to_left_poke']<=10)&(df2['Event']=='rest_starts'), 'Event']=np.nan
    events['rest_starts']=df2[df2.Event=='rest_starts'].index.values
    
    df3=df2.copy()
    # df3[['Centroid_X (cm)', 'Centroid_Y (cm)', 'Speed', 'Distance_to_left_poke']].fillna(method='ffill', inplace=True)
    df3=peri_event_time(df2.drop(columns='Event'), pd.DataFrame(df2['Event']), -20,20,2,all_trials=False)
    df3=df3[(df3['Event']!='exploratory_right_poke')&(df3['Event']!='mistake_right_poke')]
    with sns.plotting_context("poster"):
        g=sns.relplot(x='Peri-event_time', y='Speed', data=df3,kind='line', hue='Event')
        # g.fig.suptitle('Functional network '+i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=0.5,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
    #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
        g.set_ylabels('Speed (cm/sec)')
        g.set_xlabels('Time (sec)')
    #plt.ylim(-0.002,0.002)
        plt.show()
        
    with sns.plotting_context("poster"):
        g=sns.relplot(x='Peri-event_time', y='Distance_to_left_poke', data=df3,kind='line', hue='Event')
        # g.fig.suptitle('Functional network '+i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=1,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
    #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
        g.set_ylabels('Distance to left poke (cm)')
        g.set_xlabels('Time (sec)')
    #plt.ylim(-0.002,0.002)
        plt.title('rest_starts not filtered')
        plt.show()
    
    position_of_event(scaled_centroids['Centroid_X (cm)'].values, scaled_centroids['Centroid_Y (cm)'].values,np.array(scaled_centroids.index),events['rest_starts'])

    del(df3)
    # plt.plot(df2['Centroid_X (cm)'],df2['Centroid_Y (cm)'],color='gray',alpha=0.5, label='Trajectory')
    # plt.scatter(df3.loc[(df3['Event']=='left_poke')&(df3['Peri-event_time']==0),'Centroid_X (cm)'],df3.loc[(df3['Event']=='left_poke')&(df3['Peri-event_time']==0),'Centroid_Y (cm)'], color='black',s=10, label='Event positions')
    # plt.legend()
    # plt.show()
    
    
    #Incorporate events to df
    df2=incorporate_events_to_continuous(df,events)
    
    #Get rid of pauses. Replace jumps in timestamps as it had never been pauses by creating a 'Time' veriable using the first timestamp and the sampling rate, and set it as index
    df2=delete_time_gaps(df2, sampling_rate).reset_index()
    
    fp_columns=[col for col in df2.columns if 'FP' in col]
    
    print('Start looping to calculate components of the whole df')
    vol_feeding_ic_mouse_list=[]
    for start in range(0,df2.index[-1], 100000):
        if start+100000<df2.index[-1]:
            end=start+100000-1
            
        else:
            end=df2.index[-1]
        df=df2.loc[start:end,]
        df.set_index('Time', drop=True, inplace=True)
        df_spectogram=spectogram(df[fp_columns])#Make spectogram
        print('Done spectogram from '+str(start)+' to '+str(end))
        # The next line is very important!! It is to align the FP column of spectogram_df with the columns of weights dataframe
        df_spectogram=df_spectogram[weights.columns.tolist()] #Align the FP column of spectogram_df with the columns of weights dataframe

        df_spectogram.index=df_spectogram.index+df.index[0]
        event_df=pd.DataFrame(df['Event'].dropna())
        
        
        #mouse_full_spectogram_df
        
        #Add events to spectogram dataframe
        # df_spectogram.reset_index(inplace=True)
        # df_spectogram.rename(columns={'index':'Time'},inplace=True)
        # event_df.reset_index(inplace=True)
        
        # reverse_merge=pd.merge_asof(event_df,df_spectogram.reset_index(), on='Time', direction='nearest')
        # df3=df_spectogram.merge(reverse_merge[['Event','index']].set_index('index'),how='left', left_index=True, right_index=True)
        
        df3=incorporate_events_to_continuous(df_spectogram,event_df, exact=False)
        #Make peri-event_time column
        # df3['Peri-event_time']=np.nan
        # for i in df3.loc[~df3['Event'].isnull(),'Time']:
        #     df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Peri-event_time']=df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Time']-i
        #     df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Event']=df3.loc[df3['Time']==i, 'Event'].values[0]
        # df3['Peri-event_time']=df3['Peri-event_time'].round(1)
        if len(df3.dropna(subset=['Event']))>0:
            df3=peri_event_time(df3.drop(columns='Event'), pd.DataFrame(df3['Event']), -20,20,1,all_trials=False)
        
        
        ## projet ICA in the spectograms for Self stim
        
            # The next line is very important!! It is to align the FP column of spectogram_df with the columns of weights dataframe
            df3=df3[(weights.columns.tolist()+['Event', 'Peri-event_time'])] #Align the FP column of spectogram_df with the columns of weights dataframe
        
            #Calculate the values for the components in the whole spectogram dataframe, using the weights, and put them in a dataframe: ic_df
            temp_df=pd.DataFrame()
            for i in weights.index:
                temp_ind_com='IC_'+str(i+1)
                temp_df[temp_ind_com]=np.dot(df3.drop(columns=['Event', 'Peri-event_time']), weights.loc[i])

            #Incorporate ic_df to full_spectogram_df
            df4=pd.concat([df3[['Event', 'Peri-event_time']],temp_df], axis=1)
        
            vol_feeding_ic_mouse_list.append(df4)
        
        print('Done '+str(start)+' to '+str(end))
    
    del(df2)
    vol_feeding_mouse_df=pd.concat(vol_feeding_ic_mouse_list)
    del(vol_feeding_ic_mouse_list)
    #Add mouseId and Day columns
    vol_feeding_mouse_df['MouseId']=mouse
    #Add spectogram dataframe to dict
    vol_feeding_df_list.append(vol_feeding_mouse_df)
    print('Done '+mouse)
vol_feeding_df=pd.concat(vol_feeding_df_list)
print('Done projecting ica to the whole voluntary feeding data!!!!')
    

vol_feeding_dfB=vol_feeding_df.copy()
vol_feeding_dfB.reset_index(drop=True, inplace=True)

#Make perievent_df by slicing the index with not NAN values in the Peri-event_time column
# full_feeding_perievent_df=vol_feeding_dfB[~vol_feeding_dfB['Peri-event_time'].isnull()].drop(columns=['Time'])
# full_feeding_perievent_df=full_feeding_perievent_df.groupby(['Event', 'Peri-event_time', 'MouseId'], as_index=False).mean()


vol_feeding_dfB=vol_feeding_dfB.groupby(['MouseId','Event', 'Peri-event_time'],as_index=False).mean()

# Plot peri-event histograms of components, to evaluate if they change with the infusion
vol_feeding_dfC=vol_feeding_dfB[(vol_feeding_dfB['Event']=='pellet_removal')|(vol_feeding_dfB['Event']=='rest_starts')]
for i in [j for j in vol_feeding_dfB.columns if 'IC_' in j]:
    with sns.plotting_context("talk"):
        g=sns.relplot(x='Peri-event_time', y=i, data=vol_feeding_dfB,kind='line', col='Event', col_order=['left_poke','pellet_removal', 'right_poke','rest_starts', 'movement_initiation'], legend=None)
        g.fig.suptitle(i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=0,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
        # g.map(plt.axvline,x=-5,ls="--",c="gray")
        # g.map(plt.axvline,x=5,ls="--",c="gray")
        g.set_titles('{col_name}', fontweight='bold')
        g.set_ylabels('IC score')
        g.set_xlabels('Time (sec)')
        plt.ylim(-0.002,0.002)
        plt.xticks([-20,-10,0,10,20])
        plt.show()

#vol_feeding_df2['Peri-event_time'].fillna(40, inplace=True)
vol_feeding_df_list=[]
ic_col_list=[col for col in vol_feeding_dfB.columns if 'IC' in col]
for mouse in vol_feeding_dfB['MouseId'].unique():
    vol_feeding_df2=vol_feeding_dfB[vol_feeding_dfB['MouseId']==mouse]
    bins_label=list(range(int(vol_feeding_df2['Time'].max())))
    bins_limits=list(range(int(vol_feeding_df2['Time'].max()+1)))
    vol_feeding_df2['Time(rounded)']=pd.cut(vol_feeding_df2['Time'],bins=bins_limits, labels=bins_label)
    vol_feeding_df2.dropna(subset=['Time(rounded)'], inplace=True)

    events_timestamps_df=vol_feeding_df2.loc[vol_feeding_df2['Peri-event_time']==0,['Event', 'MouseId', 'Time(rounded)']]
    vol_feeding_df2=vol_feeding_df2.groupby(['MouseId', 'Time(rounded)'],as_index=False)[ic_col_list].median().dropna()

    vol_feeding_df3=vol_feeding_df2.merge(events_timestamps_df,how='left', on=['MouseId', 'Time(rounded)'], validate='one_to_one')
    vol_feeding_df3['Time(rounded)']=vol_feeding_df3['Time(rounded)'].astype(int)
    
    event_number=1
    for i in vol_feeding_df3.loc[~vol_feeding_df3['Event'].isnull(),'Time(rounded)']:
        vol_feeding_df3.loc[(vol_feeding_df3['Time(rounded)']>=(i-50))&(vol_feeding_df3['Time(rounded)']<=(i+50)), 'Peri-event_time']=vol_feeding_df3.loc[(vol_feeding_df3['Time(rounded)']>=(i-50))&(vol_feeding_df3['Time(rounded)']<=(i+50)), 'Time(rounded)']-i
        vol_feeding_df3.loc[(vol_feeding_df3['Time(rounded)']>=(i-50))&(vol_feeding_df3['Time(rounded)']<=(i+50)), 'Event']=vol_feeding_df3.loc[vol_feeding_df3['Time(rounded)']==i, 'Event'].values[0]
        vol_feeding_df3.loc[(vol_feeding_df3['Time(rounded)']>=(i-50))&(vol_feeding_df3['Time(rounded)']<=(i+50)), 'Event_number']=event_number
        event_number=event_number+1
    vol_feeding_df3['Peri-event_time']=vol_feeding_df3['Peri-event_time'].round()
    vol_feeding_df_list.append(vol_feeding_df3)
vol_feeding_dfC=pd.concat(vol_feeding_df_list, ignore_index=True)
vol_feeding_dfC['Event'].fillna('No_feeding', inplace=True)

#Make perievent_df by slicing the index with not NAN values in the Peri-event_time column
full_feeding_perievent_df=vol_feeding_dfC[~vol_feeding_dfC['Peri-event_time'].isnull()].drop(columns=['Time(rounded)'])
full_feeding_perievent_df=full_feeding_perievent_df.groupby(['MouseId','Event', 'Peri-event_time','Event_number'], as_index=False).mean()

# Plot peri-event histograms of components, to evaluate if they change with the infusion
for i in [j for j in full_feeding_perievent_df.columns if 'IC_' in j]:
    with sns.plotting_context("poster"):
        g=sns.relplot(x='Peri-event_time', y=i, data=full_feeding_perievent_df,kind='line', hue='Event')
        g.fig.suptitle('Functional network '+i, y=1.05, fontsize='large', fontweight='bold')
        g.map(plt.axhline,y=0,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="gray")
    #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
        g.set_ylabels('IC score')
        g.set_xlabels('Time (sec)')
    #plt.ylim(-0.002,0.002)
        plt.show()


for i in [j for j in full_feeding_perievent_df.columns if 'IC_' in j]:
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='Peri-event_time', y=i, data=full_feeding_perievent_df, hue='Event', legend=False)
        # g.set_titles('{col_name}', fontsize='xx-large', fontweight='bold')
        # g.map(plt.axhline,y=0,ls="--",c="black")
        # g.map(plt.axvline,x=0,ls="--",c="black")
        # plt.show()
        sns.despine(top=True,right=True)
        plt.title('Functional network '+i.replace('IC_',''), fontsize=30)
        plt.axhline(y=0,ls="--",c="black")
        plt.axvline(x=0,ls="--",c="gray")
        plt.ylabel('')
        plt.xlabel('Time (sec)', fontsize=30)
        plt.yticks(fontsize=30)
        plt.xticks([-20,-10,0,10,20], fontsize=30)
        plt.subplots_adjust(bottom=0.3)
        # plt.savefig('VoluntaryFeeding_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()
        
full_feeding_perievent_df_melt=full_feeding_perievent_df.melt(id_vars=['MouseId', 'Event', 'Peri-event_time'], var_name='Source', value_name='Score')
g=sns.relplot(x='Peri-event_time', y='Score', data=full_feeding_perievent_df_melt,kind='line', col='Source')
g.fig.suptitle('Pellet_removal', y=1.05, fontsize='large', fontweight='bold')
g.map(plt.axhline,y=0,ls="--",c="black")
g.map(plt.axvline,x=0,ls="--",c="gray")
g.set_titles('{col_name}', fontsize='xx-large', fontweight='bold')
g.set_ylabels('IC score')
g.set_xlabels('Time (sec)')
#plt.ylim(-0.002,0.002)
plt.show()

#Until here is the processing of the data, next cell is for training models and make predictions





#%%
### Project the components in the VTA self stimulation data ###

directory='C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/Multi-sites project/VTA_opto_stimulation/Self stim/Self stim over time with FEDs/Ephy&photometry/Nex_files'
os.chdir(directory)

SelfStim_full_df_dict={} # this dict will have dataframes for spectograms of each whole recording
for file in glob('*.nex5'): 
    mouse=set_variable_from_list_and_filename(file, mice_list)#Define mouse variable based on file name
    region_channels_dict=choose_region_chanels_dict() #Function to choose the right dictionary according to the mouse
    df,events,sampling_rate=pull_variables_from_nex (file, ['Self_stim']) #Read nex file and pull df of LFP, events and sampling rate
     
    #Create sham_left_poke using the make_sham_events function
    events.update(make_sham_events(events, df.index, events, 8,8,12,12))
    
    # Remove noise and plot first channel to check removal of noise
    df=delete_artifacts(dataframe=df,value=0.9, quorum=0, plot=0)
    
    ## Incorporate infusion events to df and Get rid of pauses. These two steps are necessary when there are gaps in the tiestamps 
    # as the spectogram function will not consider gaps, leading to missaligments between spectograms and events if these steps are ommited.
    #Incorporate infusion events to df
    df2=incorporate_events_to_continuous(df, events, exact=True) 
    #Get rid of pauses. Replace jumps in timestamps as it had never been pauses by creating a 'Time' veriable using the first timestamp and the sampling rate, and set it as index
    df2=delete_time_gaps(df2, sampling_rate)
    
    #Make spectograms and event_dataframe
    fp_colulmns=[col for col in df2.columns if 'FP' in col]
    SelfStim_mouse_spectogram_df=spectogram(df2[fp_colulmns])
    event_df=pd.DataFrame(df2['Event'].dropna())
    
    # #Add events to spectogram dataframe
    df3=incorporate_events_to_continuous(SelfStim_mouse_spectogram_df,event_df,exact=False)
       
    #Make peri-event_time column
    df3=peri_event_time(df3.drop(columns='Event'), pd.DataFrame(df3['Event']), -20,20,1,all_trials=False)
   
    # for i in df3.loc[~df3['Event'].isnull(),'Time']:
    #     df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Peri-event_time']=df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Time']-i
    #     df3.loc[(df3['Time']>=(i-20))&(df3['Time']<=(i+20)), 'Event']=df3.loc[df3['Time']==i, 'Event'].values[0]
    # df3['Peri-event_time']=df3['Peri-event_time'].round(1)
    #Add mouseId and Day columns
    df3['MouseId']=mouse
    #Add spectogram dataframe to dict
    SelfStim_full_df_dict.update({mouse:df3})
    print('Done '+mouse)



# # Peri-event spectogram for the NAc to validate that the timestamps of the events are correct
# SelfStim_perievent_df=df3[~df3['Peri-event_time'].isnull()]
# drop_list=[col for col in SelfStim_perievent_df.columns if 'BLA' in col or 'PFC' in col or 'NAc' in col or 'IC_' in col]
# NAc_df=SelfStim_perievent_df.drop(columns=drop_list)
# NAc_df.drop(columns=['Time'], inplace=True)
# for mouse in NAc_df['MouseId'].unique():
#     for day in NAc_df['Day'].unique():
#         for event in  NAc_df['Event'].unique():
#             df_i=NAc_df[NAc_df['Event']==event]
#             df_i.drop(columns=['Event'], inplace=True)
#             df_i=df_i.groupby(['Peri-event_time']).mean()
#             sns.heatmap(df_i.T, robust=True, center=0)
#             # plt.title(mouse+'_'+day+'_'+infusion)
#             plt.show()

SelfStim_full_df=pd.concat(SelfStim_full_df_dict.values(), ignore_index=True)
# The next line is very important!! It is to align the FP column of spectogram_df with the columns of weights dataframe
SelfStim_full_df=SelfStim_full_df[(weights.columns.tolist()+['Event', 'Peri-event_time', 'MouseId'])] #Align the FP column of spectogram_df with the columns of weights dataframe
# SelfStim_full_df=SelfStim_full_df[(weights.columns.tolist()+['Time','Event', 'Peri-event_time', 'MouseId'])] #Align the FP column of spectogram_df with the columns of weights dataframe


## projet ICA in the spectograms for Self stim
#Calculate the values for the components in the whole spectogram dataframe, using the weights, and put them in a dataframe: ic_df
SelfStim_ic_df=pd.DataFrame()
for i in weights.index:
    ss_ind_com='IC_'+str(i+1)
    SelfStim_ic_df[ss_ind_com]=np.dot(SelfStim_full_df.drop(columns=['Event', 'Peri-event_time', 'MouseId']), weights.loc[i])

#Incorporate ic_df to full_spectogram_df
SelfStim_full_df=pd.concat([SelfStim_full_df,SelfStim_ic_df], axis=1)

#Make perievent_df by slicing the index with not NAN values in the Peri-event_time column
# SelfStim_perievent_df=SelfStim_full_df[~SelfStim_full_df['Peri-event_time'].isnull()]

#Center dataframe to baseline for each mouse
SelfStim_perievent_df=center_perievent_df_to_baseline_of_each_mouse(SelfStim_full_df, -20, -15)

fp_cols=[col for col in SelfStim_perievent_df.columns if 'NAc' in col or 'BLA' in col or 'vHip' in col or 'PFC' in col]
SelfStim_perievent_df.drop(columns=fp_cols, inplace=True)

# SelfStim_perievent_df2=SelfStim_perievent_df.groupby(['MouseId','Event','Peri-event_time'], as_index=False)['IC_1', 'IC_2','IC_3', 'IC_4', 'IC_5', 'IC_6', 'IC_7', 'IC_8', 'IC_9', 'IC_10'].mean()
SelfStim_perievent_df3=SelfStim_perievent_df.melt(id_vars=['Peri-event_time', 'MouseId','Event'], var_name='Engram', value_name='Score')
SelfStim_perievent_df3['Engram']=SelfStim_perievent_df3['Engram'].str.replace('IC_', 'Engram ')
# Plot peri-event histograms of components, to evaluate if they change with the infusion

    
# os.chdir('C:/Users/casey.e/Box/Kravitz Lab Box Drive/Eric/OptoDBS2022/Poster/VTA_SelfStim')
# with sns.plotting_context("poster"):
#     g=sns.relplot(x='Peri-event_time', y='Score', data=SelfStim_perievent_df2, kind='line', col='Engram')
#         # g.fig.suptitle('Engram '+i.replace('IC_',''))
#     g.map(plt.axhline,y=0,ls="--",c="black")
#     g.map(plt.axvline,x=0,ls="--",c="gray")
#     g.set_titles('{col_name}'.replace('IC_','Engram ' ))
#     g.set_ylabels('Activity score')
#     g.set_xlabels('Time (sec)')
#     plt.ylim(-0.003,0.003)
#     plt.yticks(rotation=35)
#     plt.xticks([-15,-10,-5,0,5,10,15], rotation=35)
#     plt.show()
#     # plt.subplots_adjust(left=0.05)
#     # plt.savefig('VTASelfStim_Scores_PeriEventHist'+'.pdf')

for i in [j for j in SelfStim_perievent_df.columns if 'IC_' in j]:
    with sns.plotting_context("poster"):
        g=sns.lineplot(x='Peri-event_time', y=i, data=SelfStim_perievent_df, hue='Event', legend=False)
        sns.despine(top=True,right=True)
        plt.title('Functional network '+i.replace('IC_',''), fontsize=30)
        plt.axhline(y=0,ls="--",c="black")
        plt.axvline(x=0,ls="--",c="gray")
        plt.ylabel('')
        plt.xlabel('Time (sec)', fontsize=30)
        plt.yticks(fontsize=30)
        plt.xticks([-20,-10,0,10,20], fontsize=30)
        # plt.subplots_adjust(bottom=0.3)
        # plt.savefig('Forced_drink_Scores_PeriEventHist_'+i+'.pdf')
        plt.show()
        
for i in [j for j in SelfStim_perievent_df2.columns if 'IC_' in j]:
    with sns.plotting_context("poster"):       
        g=sns.relplot(x='Peri-event_time', y=i, data=SelfStim_perievent_df2,kind='line', col='MouseId')
        g.set_titles('{col_name}', fontsize='xx-large', fontweight='bold')
        g.map(plt.axhline,y=0,ls="--",c="black")
        g.map(plt.axvline,x=0,ls="--",c="black")
        plt.show()

# #Repeated measures anova
# #Prepare data, slice peri-event_time between -5 and 5
# rm_df=SelfStim_perievent_df2.loc[(SelfStim_perievent_df2['Peri-event_time']>-10.5)&(SelfStim_perievent_df2['Peri-event_time']<5.5)]
# rm_df['Peri-event_time']=SelfStim_perievent_df2['Peri-event_time'].round()
# rm_df=rm_df.groupby(['Peri-event_time', 'MouseId', 'Engram'], as_index=False)['Score'].mean()
# g=sns.relplot(x='Peri-event_time', y='Score', data=rm_df, kind='line', col='Engram')
# plt.show()

# for engram in rm_df['Engram'].unique():
#     df=rm_df[rm_df['Engram']==engram]
#     aov = pg.rm_anova(dv='Score', within='Peri-event_time',
#                   subject='MouseId', data=df,correction=True, detailed=True,
#                   effsize="ng2")
#     print(engram, aov.loc[0,'p-GG-corr'].round(3))
    
#Repeated measures anova: try only three times (-10 to -5, -5 to 0, and 0 to 5)
rm_df=SelfStim_perievent_df3.loc[(SelfStim_perievent_df3['Peri-event_time']>-10.5)&(SelfStim_perievent_df3['Peri-event_time']<5.5)]
labels=['baseline','pre-stim','post-stim']
limits=[-6,-3,0,3]
rm_df['Period']=pd.cut(rm_df['Peri-event_time'], bins=limits, labels=labels)
rm_df=rm_df.groupby(['MouseId','Period', 'Engram'], as_index=False)['Score'].mean()
g=sns.relplot(x='Period', y='Score', data=rm_df, kind='line', col='Engram')
plt.show()

for engram in rm_df['Engram'].unique():
    df=rm_df[rm_df['Engram']==engram]
    aov = pg.rm_anova(dv='Score', within='Period',
                  subject='MouseId', data=df,correction=True, detailed=True,
                  effsize="ng2")
    # print(engram, aov.loc[0,'p-GG-corr'].round(3))
    print(engram, aov.loc[0,'p-unc'].round(3))


# SelfStim_perievent_df=SelfStim_perievent_df.groupby(['Event', 'Peri-event_time', 'MouseId'], as_index=False).mean()
# #Center scores to the baseline (from -15 to -10sec) of each mouse
# normalized_list=[]
# for mouse in SelfStim_perievent_df['MouseId'].unique():
#     df=SelfStim_perievent_df[SelfStim_perievent_df['MouseId']==mouse]
#     for col in df.columns:
#         if 'IC_' in col:
#             mean=df.loc[(df['Peri-event_time']>-15)&(df['Peri-event_time']<-10), col].mean()
#             # std=df.loc[(df['Peri-event_time']>-10)&(df['Peri-event_time']<-7), col].std()
#             df[col]=(df[col]-mean)
#     normalized_list.append(df)
# SelfStim_perievent_df=pd.concat(normalized_list, ignore_index=True)

# # Plot peri-event histograms of components, to evaluate if they change with the infusion
# for i in [j for j in SelfStim_perievent_df.columns if 'IC_' in j]:
#     g=sns.relplot(x='Peri-event_time', y=i, data=SelfStim_perievent_df,kind='line')
#     g.fig.suptitle('VTA self stim_'+i, y=1.05, fontsize='large', fontweight='bold')
#     g.map(plt.axhline,y=0,ls="--",c="black")
#     g.map(plt.axvline,x=0,ls="--",c="gray")
#     #g.set_titles('{col_name}_{row_name}', fontsize='xx-large', fontweight='bold')
#     g.set_ylabels('IC score')
#     g.set_xlabels('Time (sec)')
#     #plt.ylim(-0.002,0.002)
#     plt.show()


## Peri-event spectogram for the NAc to validate that the timestamps of the events are correct
SelfStim_perievent_df2=SelfStim_perievent_df[~SelfStim_perievent_df['Peri-event_time'].isnull()]
drop_list=[col for col in SelfStim_perievent_df2.columns if 'NAc' in col or 'PFC' in col or 'vHip' in col or 'IC_' in col]
NAc_df=SelfStim_perievent_df2.drop(columns=drop_list)
NAc_df.drop(columns=['Time'], inplace=True)
for mouse in NAc_df['MouseId'].unique():
    # for day in NAc_df['Day'].unique():
    for event in  NAc_df['Event'].unique():
        df_i=NAc_df[(NAc_df['MouseId']==mouse)&(NAc_df['Event']==event)]
        df_i.drop(columns=['MouseId', 'Event'], inplace=True)
        df_i=df_i.groupby(['Peri-event_time']).mean()
        sns.heatmap(df_i.T, robust=True, center=0)
        plt.title(mouse+'_'+'_'+event)
        plt.show()
        
        

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix,classification_report,roc_auc_score

SelfStim_full_dfC=SelfStim_full_df.drop(columns=fp_cols)
#Smooth vol_feeding_dfC
vol_feeding_dfC[[j for j in vol_feeding_dfC.columns if 'IC_' in j]]=vol_feeding_dfC[[j for j in vol_feeding_dfC.columns if 'IC_' in j]].rolling(20).mean()
vol_feeding_dfC.dropna(inplace=True, subset=[j for j in vol_feeding_dfC.columns if 'IC_' in j])
#Make clasificatory columns
SelfStim_full_dfC=SelfStim_full_dfC.loc[(SelfStim_full_dfC['Peri-event_time']>=-1)&(SelfStim_full_dfC['Peri-event_time']<=0),]
SelfStim_full_dfC.loc[SelfStim_full_dfC['Event']=='Self_stim', 'Seeking']=1
SelfStim_full_dfC['Seeking'].fillna(0, inplace=True)

ic_cols=[col for col in SelfStim_full_dfC.columns if 'IC_' in col]
# ic_cols.remove('IC_6')
# ic_cols.remove('IC_10')
X=SelfStim_full_dfC[ic_cols].values
y=SelfStim_full_dfC['Seeking'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_probs=logreg.predict_proba(X_test)[:,1]
y_pred=logreg.predict(X_test)
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Plot tpr against fpr
fig,ax=plt.subplots()
plt.plot([0,1],[0,1],'k--', linewidth=2)
plt.plot(fpr, tpr, linewidth=2)
plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
plt.title('ROC Curve for VTA stimulation seeking', fontsize=18, fontweight='bold', y=1.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
# plt.hlines(tpr[index],xmin=0,xmax=fpr[index], linestyle='--', color='gray')
# plt.vlines(fpr[index],ymin=0,ymax=tpr[index], linestyle='--', color='gray')
plt.show()




#select the minumun threshold gor a tpr higher than 80%
index=(pd.Series(tpr)[pd.Series(tpr)>0.75]).index[0]
print('Treshold:',thresholds[index],', True positive rate: ', tpr[index], ', False positive rate: ', fpr[index])

#Calculate predicted values based on the chosen treshlod
y_pred=np.array((pd.Series(y_pred_probs)>thresholds[index])*1)
# Calculate roc_auc_score
print('ROC_AUC_Score: ', roc_auc_score(y_test, y_pred_probs))
# Calculate the confusion matrix
print(pd.DataFrame(columns=['Predicted:Not_feeding','Predicted:Feeding'],index=['Observed:Not_feeding','Observed:Feeding'], data=confusion_matrix(y_test, y_pred)))
# Calculate the classification report
print(classification_report(y_test, y_pred))
#print false positive rate
print('False positive rate: ', fpr[index])


print(clf.score(X_test,y_test))


names=SelfStim_full_dfC[ic_cols].columns
reg_coef=abs(logreg.coef_[0])

fig,ax=plt.subplots()
plt.bar(names, reg_coef)
plt.xlabel('Functional Network', fontsize=15, fontweight='bold')
plt.ylabel('Coeficient', fontsize=15, fontweight='bold')
plt.title('Coeficient values', fontsize=18, fontweight='bold', y=1.05)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15, rotation=40)
plt.hlines(0.5,xmin=-0.6,xmax='IC_10', linestyle='--', color='black', linewidth=2)
plt.show()
