#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:48:18 2017

@author: alex
"""
import numpy as np
import mne 
import matplotlib.pyplot as plt


def get_eventfile(raw,event_file):
    
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
#Everytime this is running it hangs up afterwards and it has to sth to do with mne.pickchannels

#trig_chan = mne.pick_channels(raw.ch_names, include=['MISC001'])#this creates an index
#API has changed. Different call below. Now this won't give me an index but changes the raw object. Undesirable.
#trig_chan = raw.pick_channels(['MISC001'])#this changes the raw object
    trig_chan=raw.info['ch_names'].index('MISC001')  #note that this method only gives FIRST index. Not problematic here.
    trig_data_uncut, times_uncut=raw.get_data(trig_chan,return_times=True)
    
    
    #plt.plot(times_uncut,trig_data_uncut.T,'k')
    #9s seems to work both for eyes open and closed
    
    # there' mess at the beginning where the diode picked up control screen etc. Stops at roughly 9s. Find index and cut beginning off.
   
    
    time_start=0
    start_idx=np.argmin(np.abs(times_uncut-time_start))
    #print 'start_index=', start_idx
    times=times_uncut[start_idx:]
    trig_data=trig_data_uncut.T[start_idx:]#transpose then you don't have to worry about plots
    trig_data=(trig_data-min(trig_data))/(max(trig_data)-min(trig_data)) #normalise
    
    # now find trigger values above thresh. Ideally we take the first derivative to get the slope of the curve. Diff will do.
    #slope=np.ediff1d(trig_data) #diff will actually create an array with 0 dim on one axis. Don't know what to make of that.
    slope=np.diff(trig_data,axis=0) 
    thresh=.1 #looks like a good choice.
    
    
    # Now we find sample points where values are above threshold:
    above_thresh=np.where(slope>thresh)#make logical vector
    above_thresh=above_thresh[0]#get logical vector out of the tuple which I think is like a cell?!
    #OK I have now all the indices of timepoints where signal is above threshold in a numpy array
    
    
    # some points are successive, in that the difference is less than an interval flash interval. Here 1 data point apart. i.e. are part of the same flash. 
    #What's the minimum interval between flashes?
    #Do the same again as above, basically. Take the difference between timepoints and reject those to close together.
    inter_time=np.diff(above_thresh) 
    isfirst=inter_time>2000 #= logical array. ISI between 4000-6000ish? So that should be more than enough
    
    
    #isfirst is one shorter than above_thresh. That's why we save the first time point separately.
    # remove all inter_times where isfirst is false
    first_trial=above_thresh[0] #save first time point because I'm doing another diff.
    
    
    # Now extract event indices and times using the logical vecor isfirst
    events=above_thresh[1:][isfirst] #remember, we can't include t1 here. We append in the next step--
    events=np.append(first_trial,events)
    event_time=times[events]
    
    # CHECK
    #200 flashes
    plt.figure()
    plt.plot(times_uncut,trig_data_uncut.T,'k')
    plt.plot(times,trig_data,'b') #ok
    plt.plot(times[1:],slope,'g') #ok
    plt.plot(times[events],slope[events],'rx')
    
    #The event list contains three columns. The first column corresponds to sample number. 
    #To convert this to seconds, you should divide the sample number by the used sampling frequency. 
    #The second column is reserved for the old value of the trigger channel at the time of transition, 
    #but is currently not in use. 
    #The third column is the trigger id (amplitude of the pulse).
    event_list=np.zeros((events.shape[0],3))
    event_list[:,0]=events+first_samp+start_idx #does this fix the time line?
    event_list[:,2]=1
    event_list=event_list.astype(np.int32)#? good enough?
    
    if event_list[1,0]-event_list[0,0]>3*sfreq:
        event_list=event_list[1:,:]
    
    event_id = {'flash': 1}
    event_color = {1: 'green'}
    
    # Sanity check---> Plot the events to get an idea of the paradigm
    mne.viz.plot_events(event_list, sfreq, first_samp, color=event_color,
                    event_id=event_id)
    
    #save eventlist
    np.save(event_file, event_list) ## Now I don't have to do this again. remove fif-extension
    
    return event_list