import pandas as pd
import numpy as np
import Features as feat
from os import path
import matplotlib.pyplot as plt


dfs = pd.read_hdf(path.join("..", "chunks_big", "dataframe_0.h5"), key = 'table')

df1 = pd.DataFrame()

for index, trip in dfs.groupby(level = ['Driver', 'Trip']):

    features = feat.Features(trip, [])
    
    # In km/h
    velocities = features.euclidean_helper() * 2.2369
    
    plt.hist(velocities, bins = np.arange(0, 100, 5))
    plt.show()

