import pandas as pd
import numpy as np

df = pd.read_hdf('dataframe.h5', key = 'table')
print(df)