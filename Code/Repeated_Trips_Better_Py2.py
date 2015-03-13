from __future__ import division
import pandas as pd
import numpy as np
from os import path, listdir
import time
from scipy.spatial.distance import cdist
import h5py
import multiprocessing as mp

DISTANCE = 50
TOLERANCE = 0.017


# Chunks (containing parts of the mega df)
chunk_path = "/scratch/vstrobel/chunks32"
matched_trips_path = "~/awesome-kagg-ml/matched"

def rotational(theta):
    # http://en.wikipedia.org/wiki/Rotation_matrix
    # Beyond rotation matrix, fliping, scaling, shear can be combined into a single affine transform
    # http://en.wikipedia.org/wiki/Affine_transformation#mediaviewer/File:2D_affine_transformation_matrix.svg
    return np.array([[-np.sin(theta), np.cos(theta)], [np.cos(theta), np.sin(theta)]])


def flip(x):
    # flip a trip if more that half of coordinates have y axis value above 0
    if np.sign(x[:,1]).sum() > 0:
        x = x.dot(np.array([[1,0],[0,-1]]))
    return pd.DataFrame(x, columns=['x', 'y'])


def rotate_trip(trip):
    # take last element
    a = trip.iloc[-1]
    # get the degree to rotate
    w0 = np.arctan2(a.y,a.x) # from origin to last element angle
    # rotate using the rotational: equivalent to rotational(-w0).dot(trip.T).T
    return np.array(trip.dot(rotational(w0)))

def compressed_trip(trip):

    diff1 = np.diff(trip.x) ** 2
    diff2 = np.diff(trip.y) ** 2
    euc_dist = np.sqrt(diff1 + diff2)

    euc_cum = np.cumsum(euc_dist)
    sol = []
    for i in range(int((euc_cum[-1])/ DISTANCE)):
        sol.append(next(j for j in range(len(euc_cum)) if euc_cum[j]-(i* DISTANCE) >= 0))
    
    sol_arr = np.array(sol, dtype = int)

    return trip.ix[sol_arr]


def similarity_trips(trips):

    to = int(time.time())
    sim = np.zeros((201, 201))

    for trip_num, trip in trips.groupby(level = ['Trip']):
        max_sim = sim[trip_num, :].max()
        for other_trip_num, other_trip in trips.groupby(level = ['Trip']):
            if (trip_num != other_trip_num) or (sim[trip_num, other_trip_num] == 0):

                if len(trip) > len(other_trip):
                    lt = trip
                    st = other_trip
                else:
                    lt = other_trip
                    st = trip

                dist = len(lt) - len(st)
                dnf = (len(st)/len(lt))

                if dnf > max_sim:
                    for i in np.random.permutation(range(0, dist, 10)):

                        new_lt = pd.DataFrame()
                        new_lt['x'] = lt.x - lt.ix[i, 'x']
                        new_lt['y'] = lt.y - lt.ix[i, 'y']
                        b = new_lt.iloc[i+len(st)]
                        beta = np.arctan2(b.y,b.x) # from origin to last element angle
                        rlt = np.array(new_lt.dot(rotational(beta)))
                        rst = np.array(st.dot(rotational(beta)))
                        tmp_sim = cdist(rlt[i:i+len(rst)], rst) # Kevin
                        # tmp_sim = cdist(np.diff(rlt[i:i+len(rst)]), np.diff(rst)) # Volker
                        current_sim = np.minimum(1.0, len(rst) * DISTANCE * TOLERANCE * (1/(tmp_sim.mean())) * dnf)

                        sim[trip_num, other_trip_num] = current_sim
                        sim[other_trip_num, trip_num] = current_sim

    return sim


def preprocessing(driver, trips):

    ls_trips = []

    for trip_num, trip in trips.groupby(level = ['Trip']):
        ls_trips.append(pd.concat([compressed_trip(flip(rotate_trip(trip)))],
                                    keys = [(driver, trip_num)],
                                    names = ('Driver', 'Trip')))

    return pd.concat(ls_trips)


def do_jobs(chunk):
    
    df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

    for driver, trips in df.groupby(level = ['Driver']):
        print(driver)
        new_trips = preprocessing(driver, trips)
        sims = similarity_trips(new_trips)
        h5f = h5py.File(matched_trips_path + 'data-{}.h5'.format(driver), 'w')
        h5f.create_dataset('dataset_{}'.format(driver), data=sims)
        h5f.close()


def main():

    chunks = listdir(chunk_path)

    jobs = []

    for chunk in chunks:

        p = mp.Process(target = do_jobs, args = (chunk, )) 
        jobs.append(p)
        p.start()


if __name__ == "__main__":
    main()
