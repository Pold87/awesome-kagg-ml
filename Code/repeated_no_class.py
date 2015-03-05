#!/usr/bin/env pypy

# Downloaded from:
# https://github.com/jimonji/telematic-competition

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_distances
from os import listdir, path
from numba import autojit


@autojit
def rotational(theta):
    # http://en.wikipedia.org/wiki/Rotation_matrix
    # Beyond rotation matrix, fliping, scaling, shear can be combined into a single affine transform
    # http://en.wikipedia.org/wiki/Affine_transformation#mediaviewer/File:2D_affine_transformation_matrix.svg
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

@autojit
def flip(x):
    # flip a trip if more that half of coordinates have y axis value above 0
    if np.sign(x[:,1]).sum() > 0:
        x = x.dot(np.array([[1,0],[0,-1]]))
    return x

@autojit
def rotate_trip(trip):
    # take last element
    a=trip.iloc[-1]
    # get the degree to rotate
    w0=np.arctan2(a.y,a.x) # from origin to last element angle
    # rotate using the rotational: equivalent to rotational(-w0).dot(trip.T).T
    return np.array(trip.dot(rotational(w0)))
 # get x,y coordinate

@autojit
def getxy(a):
    (d,w)=a
    return d*np.cos(w),d*np.sin(w)
# get r,w coordinate

@autojit
def getwd(a):
    [x,y]=a
    return np.hypot(x,y), np.arctan2(y,x)

@autojit
def update_trips(tripl, trip, tripname, tripcounter, tripother):
    if len(tripl)==0:
        tripl[tripname]=trip
        tripcounter[tripname]=0
        tripother[tripname]=[]
    else:
        for t in tripl:
            if sametrip(tripl[t],trip):
                tripcounter[t] += 1
                tripother[t].append(tripname)
                for xx in tripother[t]:
                    tripcounter[xx] = tripcounter[t]
                return tripl, tripcounter, tripother
        tripl[tripname] = trip
        tripcounter[tripname] = 0
        tripother[tripname] = []
    return tripl, tripcounter, tripother

@autojit
def getdd(tx,ty):
    """
    calculate distances and pad shortest array
    pad shortest array with it's last value
    """
    gap = tx.shape[0]-ty.shape[0]
    if gap > 0:
        ty = np.pad(ty, ((0, gap), (0, 0)), 'edge')
    elif gap <0:
        tx = np.pad(tx, ((0, -gap), (0, 0)), 'edge')
    else:
        pass
    # use any distance metric that you would like
    return paired_distances(tx, ty, metric='l2').sum()

@autojit
def sametrip(tx, ty):
    mm = int(tx.shape[0] * threshold)
    txx = np.vstack((tx[mm:, :], tx[-mm:, :]))
    dd = getdd(tx, txx)
    return getdd(tx, ty) <= dd

@autojit
def check_trips(tripcounter, tripname):
    return tripcounter[tripname] > 2

def draw(driver):
    path = dataPath
    drivers = os.listdir(path)
    print('driver',driver)
    trips=os.listdir(path+'/'+driver)
    tripl = {} # trip name -> trip data
    tripc = {} # trip name -> number of same trips
    tripo = {} # trip name -> list of names of other same trips
    for trip in trips:
        t1 = pd.read_csv(path+'/'+driver+'/'+trip)
        t1r = flip(rotate_trip(t1))
        tripl, tripc, tripo = update_trips(tripl, t1r, trip, tripc, tripo)
    for trip in trips:
        t1 = pd.read_csv(path+'/'+driver+'/'+trip)
        flip(rotate_trip(t1))
        if check_trips(tripc, trip):
            df_submission.loc[str(driver) + '_' + str(trip), 'prob'] = 1


mypath = "../drivers"

drivers_path = "../drivers"
mydrivers = listdir(drivers_path)

dataPath = mypath
threshold = float(0.05)
df_submission = pd.read_csv('submission-6.csv')

for driver in mydrivers:
    draw(driver)

df_submission.to_csv('submission_processed_6.csv', index = False)