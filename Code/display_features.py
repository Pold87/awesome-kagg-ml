import pandas as pd
import numpy as np

features = pd.read_hdf("/scratch/vstrobel/features_opti_32/features_0.h5", key = 'table')


print(features.head())

