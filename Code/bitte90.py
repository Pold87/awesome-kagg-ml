from __future__ import division
import sys
import collections
import itertools
from scipy.stats import mode
import pandas as pd
import numpy as np
from os import listdir, path
import multiprocessing as mp
import h5py

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function
            
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    
    def __init__(self, n_neighbors=1, max_warping_window=50, subsample_step=20):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
            
        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
        self.x = x
        self.l = l
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxint * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in xrange(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in xrange(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in xrange(1, M):
            for j in xrange(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window 
        return cost[-1, -1]
    
    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        
        y : array of shape [n_samples, n_timepoints]
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            p = ProgressBar(shape(dm)[0])
            
            for i in xrange(0, x_s[0] - 1):
                for j in xrange(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
                    p.animate(dm_count)
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            dm_size = x_s[0]*y_s[0]
            
            p = ProgressBar(dm_size)
        
            for i in xrange(0, x_s[0]):
                for j in xrange(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)
        
            return dm
        
    def predict(self, x):
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels 
              (2) the knn label count probability
        """
        
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


##################################################


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
    a=trip.iloc[-1]
    # get the degree to rotate
    w0=np.arctan2(a.y,a.x) # from origin to last element angle
    # rotate using the rotational: equivalent to rotational(-w0).dot(trip.T).T
    return np.array(trip.dot(rotational(w0)))


def do_job(i, chunk):

    df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

    for driver, trips in df.groupby(level = ['Driver']):


        print('driver is   ')

        print(driver)

        sims = similarity_trips(trips)
        h5f = h5py.File(matched_trips_path + 'data-{}.h5'.format(driver), 'w')
        h5f.create_dataset('dataset_{}'.format(driver), data=sims)
        h5f.close()


def similarity_trips(trips):




    m = KnnDtw(n_neighbors = 1, max_warping_window=50, subsample_step=10)

    sim = np.zeros((201, 201))
    
    for trip_num, trip in trips.groupby(level = ['Trip']):
        print(trip_num)

        if int(trip_num) != 1: continue

        for other_trip_num, other_trip in trips.groupby(level = ['Trip']):
            
            if (int (trip_num) != int(other_trip_num)) or (sim[trip_num, other_trip_num] == 0):
            
                print(other_trip_num)
    
                trip1 = flip(rotate_trip(trip))
                trip2 = flip(rotate_trip(other_trip))


                distance = m._dtw_distance(trip1.values,
                                           trip2.values,
                                           d = lambda x,y: np.linalg.norm(x-y))
                

                sim[trip_num, other_trip_num] = distance
                sim[other_trip_num, trip_num] = distance

                if int(other_trip_num) == 200: break


        break

    print(np.min(sim))
    non_zero_mask = sim[1, :] != 0
    first_row = sim[1, :]
    non_zero = first_row[non_zero_mask]

    sim = 1 - (np.max(non_zero) / sim)

    print(sim)

    first_row = sim[1, :]

    print(zip(np.arange(201), first_row))

    return sim

        
        
# Chunks (containing parts of the mega df)
chunk_path = "/scratch/vstrobel/chunks32_small/"
matched_trips_path = "/scratch/vstrobel/matched_dtw/"
chunks = sorted(listdir(chunk_path))

def main():

    jobs = []

    print(chunks[:1])

    for i, chunk in enumerate(chunks[:1]):
        p = mp.Process(target = do_job, args = (i,chunk, ))
        jobs.append(p)
        p.start()

if __name__ == "__main__":
    main()
