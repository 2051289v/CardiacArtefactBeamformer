#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:23:40 2017
Little maxfilter routine using default parameters except st_correlation 90
rather than 98 and st_duration because those are the parameters Chris used.
I hope these correspond - looked at flash with Elekta vs this and it seemed
pretty much the same.
@author: alex
"""
# preparing everything:
import os.path as op
import mne
from os import makedirs
import re

mne.set_log_level('WARNING')

#
# 1421 1431 are the worst affected MAGs of slow artefact.
# Can this be harnessed? ICA?
subject = '0001'
recdate = '20170801'  #'20171215' '20180302' 
# where are the data?
proj_path = '/projects/MINDLAB2016_MEG-ERG-flash/'
output_path = op.join(proj_path, 'scratch', 'maxfilter', subject, recdate)
if not op.exists(output_path):
    makedirs(output_path)

data_path = op.join(proj_path, 'raw',
                    subject, recdate +    
# 20180302:                   
#                    '_000000/MEG/001.audioOnly/files/')
#                    '_000000/MEG/003.audioVisual/files/')
#                   '_000000/MEG/005.flash_longerISI/files/')
# bads = ['MEG1143', 'MEG0121', 'MEG0441']
# 20171215:
#                  ' _000000/MEG/003.eyes_open_rest/files/')
#                  ' _000000/MEG/002.eyes_closed_rest/files/')
#                  '_000000/MEG/001.empty_room_2min/files/')
                  '_000000/MEG/004.flash_open/files/')
#bads = ['MEG0121', 'MEG1143', 'MEG0712']  # noisy, flat, spiky, respectively
bads = ['MEG0121', 'MEG1143', 'MEG1423', 'MEG1433']  #excluded by ELEKTA

task_name = re.search('(?<=MEG/00\d\.)\w+', data_path).group(0)
print task_name
raw_fname = data_path + task_name + '.fif'      


raw = mne.io.read_raw_fif(raw_fname)
if bads is not None:
    raw.info['bads'] = bads

print 'Done with preliminaries.'

# %% try maxwell filter
win = 30
corr = 0.90
raw = mne.preprocessing.maxwell_filter(raw, origin='auto', int_order=8,
                                       ext_order=3, calibration=None,
                                       cross_talk=None, st_duration=win,
                                       st_correlation=corr,
                                       coord_frame='head', destination=None,
                                       regularize='in', ignore_ref=False,
                                       bad_condition='error', head_pos=None,
                                       st_fixed=True, st_only=False,
                                       mag_scale=100.0)
maxfilt_fname = output_path + '/' + task_name + '-MNE_tsss_win' + str(win) \
                + '_corr' + str(int(corr * 100)) + '.fif'
raw.save(maxfilt_fname, overwrite=True)
