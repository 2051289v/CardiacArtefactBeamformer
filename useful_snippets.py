#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:20:21 2018

@author: alexandrav
"""
import numpy as np


# find the smallest difference between the closest events in 2 time series
def findMinDiff(arr,arr2):
    minDiffs=[];
    for x in arr:
        minDiffs.append(min(abs(arr2-x)))
    return minDiffs


def trimmean(arr, percent, axis):
    n = len(arr)
    k = int(round(n*(float(percent)/100)/2))
    return np.mean(arr[k+1:n-k], axis=axis)


def find_subset(vec,lowerbound,upperbound):
    idx=np.arange(np.argmin(abs(vec-lowerbound)),np.argmin(abs(vec-upperbound))+1)
    return idx