#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:53:40 2018

@author: alexandrav
"""
from useful_snippets import find_subset 
import os.path as op
import numpy as np
import mne
from mne.beamformer import (make_lcmv, apply_lcmv_epochs, lcmv, apply_lcmv)
import matplotlib.pyplot as plt
from nilearn.plotting import (plot_stat_map, plot_glass_brain)
from nilearn.image import index_img
from get_eventfile_fun import get_eventfile
import glob
import re
from os import makedirs
from cycler import cycler
get_ipython().run_line_magic('matplotlib', 'qt')
mne.set_log_level('WARNING')


## PRELIMINARIES
# set specific subject/file info
subject = '0016'
recdate = '20171215'
whichrun = '004.flash_open'
t1='fsaverage' #subject if has T1
bads=['MEG0121','MEG1143']

MaxFil = True
megtype='mag' #'mag','grad',True
hp, lp = 1, 40
trial='pulse' # 'flash' or 'pulse'

# set paths
proj_path = 'pathToFlashProject'
output_path = op.join(proj_path,'scratch')
subjects_dir = op.join(output_path, 'subjects_dir') #freesurfer reconstructions
data_path = op.join(proj_path, 'raw', subject, recdate + '_000000', 'MEG',
                    whichrun, 'files')

# get file names
task_name = re.search('(?<=MEG/00\d\.)\w+', data_path).group(0)
raw_fname = list(sorted(glob.glob(op.join(data_path , '*.fif'))))[0]
t1_fname = op.join(subjects_dir, t1, 'mri/T1.mgz')
bem_fname = op.join(subjects_dir, subject, 'bem', subject + '-inner_skull-bem-sol.fif')
trans_fname = op.join(output_path, 'src', subject, recdate, subject + '-trans.fif')
fwd_fname = op.join(output_path, 'src', subject, recdate, 'vol-fwd.fif')

# set folder and file names for auxiliary data:
aux_folder = op.join(output_path, 'src', subject, recdate, task_name, 'aux')
if not op.exists(aux_folder):
    makedirs(aux_folder)
event_file = op.join(aux_folder, 'flash_events.npy')
eog_file = op.join(aux_folder, 'blinks.npy')  
vp_file = op.join(aux_folder, 'vp_events.npy')  

# if using maxfiltered data make sure MEG data is saved somewhere else
maxfilter_path = op.join(output_path, 'maxfilter', subject)
if MaxFil:
    raw_fname = sorted(glob.glob(op.join(
            maxfilter_path, '*{}*.fif').format(task_name)), key=len)[0]
    task_name += 'MaxFil'

# now create sensor-specific folders and filenames   
if megtype is True:
    output_folder = op.join(output_path, 'src', subject, recdate, task_name, 'allMEG')
else:
    output_folder = op.join(output_path, 'src', subject, recdate, task_name, megtype)

evoked_folder = op.join(output_folder, 'evoked')
if not op.exists(evoked_folder):
    makedirs(evoked_folder)

# event information for epochs:
#ecg_fname = op.join(aux_folder,'ECG_chan.fif')
epochs_fname = op.join(evoked_folder, 'flash_epochs_MN_' + str(hp) + '-'
                          + str(lp) + 'Hz-epo.fif')
vp_epochs_fname = op.join(evoked_folder, 'vp_epochs_MN_' + str(hp) + '-'
                          + str(lp) + 'Hz-epo.fif')

event_id, tmin, tmax = 1, -.5, .5  
pulse_id=999
baseline = (-.2, -0.05)

print 'Done with preliminaries.'

## Compute source information
## Compute or load forward solution
if not op.isfile(fwd_fname):
    raw = mne.io.read_raw_fif(raw_fname)
    # use a volume source grid
    src = mne.setup_volume_source_space(subject, pos=10., mri=t1_fname, 
                                        bem=bem_fname, subjects_dir=subjects_dir)
    # make leadfield
    fwd = mne.make_forward_solution(raw.info, trans=trans_fname, src=src,
                                    bem=bem_fname, meg=True, eeg=False, n_jobs=1)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=False)            
else: 
    fwd = mne.read_forward_solution(fwd_fname)

print 'got forward model.'

##  Preprocessing: load raw data and find events and blinks, filter, epoch, decimate
if not op.exists(epochs_fname):
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
#    raw.plot(duration=4, n_channels=20)  
    print 'estimated rank = ' + str(raw.estimate_rank())
    if bads is not None:
        raw.info['bads'] = bads
    
    if op.isfile(event_file):
        events = np.load(event_file)
        print 'load event file ', event_file, '...'
    else:
        print 'event file ', event_file, ' not found, retrieving event data...'
       
        events=get_eventfile(raw,event_file)
        #execfile(script_path + 'get_eventfile.py')
    
    print 'got events'
    
    if op.isfile(eog_file):
        blinks = np.load(eog_file)
        print 'Loading eog event file ' + eog_file + '...'
    else:
        blinks = mne.preprocessing.find_eog_events(raw)
        np.save(eog_file,blinks)
    
    print 'got blinks'
    
    n_blinks = len(blinks)
    onset = (blinks[:, 0] - raw.first_samp)/ raw.info['sfreq'] - 0.2
    duration = np.repeat(0.4, n_blinks)
    description = ['bad eye'] * n_blinks
    annotations = mne.Annotations(onset, duration, description)
    raw.annotations = annotations
    
    raw.pick_types(meg=megtype, eog=False, emg=False, exclude='bads',
                   include=['ECG001'])
    
    raw.filter(hp, lp, h_trans_bandwidth='auto', filter_length='auto',
                   phase='zero', n_jobs=1, fir_design='firwin')
    
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=baseline, reject=None,
                        reject_by_annotation=True, proj=False, detrend=1)
    epochs.decimate(decim=5, offset=0, verbose=None)
    epochs.save(epochs_fname)
else:
    print 'Loading epochs...' + epochs_fname
    epochs = mne.read_epochs(epochs_fname,  preload=True, proj=False)
    
## Calculate evoked responses
evoked = epochs.average()
# get ECG epochs if required
if trial is 'pulse':
    if op.isfile(vp_epochs_fname):
        vp_epochs = mne.read_epochs(vp_epochs_fname, proj=False)
        print 'loaded pulse artefact epochs...'
    else:
        print 'calculating pulse epochs...'
        # set up file
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.info['bads'] = bads
        raw.pick_types(meg=megtype, exclude='bads', include=['ECG001','MISC001'])
        raw.filter(hp, lp, h_trans_bandwidth='auto', filter_length='auto',
                   phase='zero', n_jobs=1, fir_design='firwin')

        # apply blink removal stuff for epoching
        pulse_id = 999
        
        blinks = np.load(eog_file)
        n_blinks = len(blinks)
        onset = (blinks[:, 0] - raw.first_samp) / raw.info['sfreq'] - 0.2
        duration = np.repeat(0.4, n_blinks)
        description = ['bad eye'] * n_blinks
        annotations = mne.Annotations(onset, duration, description)
        raw.annotations = annotations
        if op.isfile(vp_file):
            vp_events = np.load(vp_file)
        else:
            # find QRS events.
            qrs_events, _, _ = mne.preprocessing.find_ecg_events(raw, pulse_id,
                                                                 ch_name='ECG001')
            # now shift to peak of pulse artefact
            vp_events = []
            for beat in qrs_events:
                samplept_adj = beat[0] - raw.first_samp
                # ca 150ms before, 150*5=750
                data_piece = raw._data[0, samplept_adj - 750: samplept_adj]
                if data_piece.any():
                    # if-loop to avoid error where not enough data at start
                    # or end of file
    
                    # find maximum in this data piece, i.e. the pulse and find
                    # how much shift is needed
                    shiftby = len(data_piece) - np.argmax(data_piece)
                    beat[0] -= shiftby
                    vp_events.append(beat)
            np.save(vp_file,vp_events)
        # Read epochs, requires more than .7 s
        vp_epochs = mne.Epochs(raw, vp_events, pulse_id, tmin, tmax,
                               baseline=baseline, reject_by_annotation=True,
                               proj=False,
                               preload=True)

        vp_epochs.decimate(decim=5, offset=0, verbose=None)
        vp_epochs.save(vp_epochs_fname)
        
    vp_evoked = vp_epochs.average()
#vp_evoked.plot(spatial_colors=True,proj=True)
#mags= mne.pick_types(epochs.info,meg='mag')


## compute sensor covariance
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.5)
noise_cov = None# mne.compute_covariance(epochs, tmin=-0.5, tmax=0)
#data_cov.plot(epochs.info)
#noise_cov.plot(epochs.info)
print 'computed covariance on all epochs...'

cond = np.linalg.cond(data_cov.data)
print 'Condition number is ' + str(cond)

## for plotting covariance:
#fig_cov,fig_svd = data_cov.plot(epochs.info)
#plt.savefig(op.join(evoked_folder, "SVD_{}_{}-{}Hz_MN_noCov.svg".format(megtype, hp,lp)), format="svg")
#xy=plt.gca().get_lines()[0].get_xydata() #it's just one line. get XY data
##xy[:,1]=np.log10(xy[:,1])
#np.save("SVD_{}_{}_{}-{}Hz_MN_noCov.npy".format(task_name, megtype, hp,lp),xy)


# Do beamformer analysis

rv = 0.001  # zero is not an option >> Britta?
reg_folder = op.join(evoked_folder, 'reg')
if not op.exists(reg_folder):
    makedirs(reg_folder)

src_folder = op.join(reg_folder, str(rv))
if not op.exists(src_folder):
    makedirs(src_folder)

# calculate or load source calc. 
fid = glob.glob(src_folder+'/{}_{}-{}Hz_MN_noCov-vl.stc'.format(trial, hp, lp))
if not fid: 
    print 'Now regularization at ' + str(rv) + '%'
    # compute spatial filter
    filters = make_lcmv(epochs.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori='max-power',  # noisecov
                        weight_norm='nai', reg=rv)
    print 'computed filter'

    # apply spatial filter:
    if trial is 'pulse':
        stc = apply_lcmv(vp_evoked, filters, max_ori_out='signed')
    else:
        stc = apply_lcmv(evoked, filters, max_ori_out='signed')
              
    stc.save(op.join(src_folder, trial
                 + '_' + str(hp) + '-'
                 + str(lp) + 'Hz_' + 'MN_noCov'), verbose=True)
   
    print 'applied filters and computed stc'
else:
    print 'Reading', fid[0]#.replace(src_folder, '', 1)
#           files = sorted(glob.glob(src_folder+'/*{}-{}*.stc'.format(hp, lp)))
    stc = mne.read_source_estimate(fid[0], subject)

## plots
# plot evoked response
fig, ax = plt.subplots(4, 1, squeeze=False, figsize=(10, 16))
x_ax = np.linspace(stc.times.min()*1000,
                   stc.times.max()*1000,
                   stc.data.shape[1])
if trial is 'pulse':
    vp_evoked.plot(gfp=False, spatial_colors=True, proj=True,axes=ax[0,0],xlim=(-200,500))
else:
    evoked.plot(gfp=False, spatial_colors=True, proj=True,axes=ax[0,0],xlim=(-200,500))

n_lines=40
ax[1,0].set_prop_cycle(cycler('color',[plt.cm.GnBu(i) for i in np.linspace(0, 1, n_lines)]))

# plot absolute magnitude of source activity for 40 most active voxels(time series)   
stc2=stc.copy()
 # take absolute values for plotting
stc2.data = np.abs(stc.data.copy())
#stc.data = np.abs(stc.data)
    
ax[1, 0].plot(x_ax, stc2.data[np.argsort(
        np.max(stc2.data, axis=1))[-40:]].T)
# -40: plots the last (highest?) 40 voxels\
ax[1, 0].set_ylabel('LCMV value')
ax[1, 0].set_xlabel('Time (ms)')
ax[1, 0].set_xlim(-200,500)#(x_ax[0], x_ax[-1])
ax[1, 0].set_title('%s %s, reg=%.3f, '
                   ' 40 max voxels' % (megtype, 'flash', rv))

#  plot brain activity at the max mimum time point in volume
vox, timep = np.unravel_index(stc.data.argmax(),
                              stc.data.shape)

img_mne = mne.save_stc_as_volume(op.join(reg_folder,
                                         'test.nii.gz'),
                                 stc2, fwd['src'])
if trial is 'pulse':
    timep = abs(stc.times + .001).argmin()
else:    
    timep = abs(stc.times - 0.108).argmin()
# plotting threshold based on activity:
thresh = np.max(stc2.data[:, timep])*0.75


#which_times=find_subset(stc.times,.105,.110)
#timep=which_times
# voxel 128.0 128.0 48.0, mm 1.0 -81.0 1.0 coords = (1,81,1)# fwd['source_rr'][vox]

# plot statistical map on volume
plot_stat_map(index_img(img_mne, timep), t1_fname,
              #cut_coords=(6.0, -32.0, 19.0),
              threshold=thresh,
              title='Amplitude at '
              't=%.3f s, reg=%.3f'
              % (stc.times[timep], rv),
              axes=ax[2, 0]
              )

ax[0, 0].set_xlabel('Time (ms)')


# plot source in glass brain:
plot_glass_brain(index_img(img_mne, timep), colorbar=True, 
                                  figure=None, axes=ax[3,0], title=None, 
                                  threshold=.01, annotate=True, 
                                  black_bg=True, cmap='viridis', alpha=0.7, 
                                  vmin=None, vmax=None, plot_abs=True, 
                                  symmetric_cbar=False, 
                                  resampling_interpolation='continuous')
# http://nilearn.github.io/modules/generated/nilearn.plotting.plot_glass_brain.html#nilearn.plotting.plot_glass_brain

fig.subplots_adjust(hspace=.25)

plt.savefig(op.join(reg_folder, str(rv),"{}_{}_{}-{}Hz_MN_noCov.svg".format(trial, megtype, hp,lp)), format="svg")

#fig2, ax2 = plt.subplots(1,1, figsize=(10, 3))
#n_lines=vp_epochs._data.shape[0]
#ax2.set_prop_cycle(cycler('color',[plt.cm.Greens(i) for i in np.linspace(0, 1, n_lines)]))
##vp_epochs.plot_image(0)
#
#ax2.plot(vp_epochs.times*1000,vp_epochs._data[:,0,:].T)
#ax2.set_xlim(-200,500)
#ax2.set_ylabel('uV')
#ax2.set_xlabel('Time (ms)')
#
#ax2.set_title('ECG')
# %%

#grads: pulse 1-300Hz maxAbs at -1.00043ms 1.1817 LCMV units 
#        flash 1-40Hz maxAbs 108.04ms,  4.26221 LCMV units  
#        flash 1-300Hz maxAbs 106.00ms,  3.1612 LCMV units    
#mags: pulse 1-300Hz maxAbs at -1.00036ms 1.89016 LCMV units 
#        flash 1-40Hz maxAbs 108.896ms,  4.66314 LCMV units 
#        flash 1-300Hz maxAbs 104.999ms,  3.69564 LCMV units   
#gradsratio = 100*1.1817/4.26221
#magsratio = 100*1.89016/4.66314
gradsratio = 100*1.1817/3.1612
magsratio = 100*1.89016/3.69564
print('mags:  %3.1f percent, grads: %3.1f percent' % (magsratio,gradsratio))
#print('{0:3.1f}'.format(0/10.0))  # '0.0'
