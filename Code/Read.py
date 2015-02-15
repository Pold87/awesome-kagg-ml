import pandas as pd
import numpy as np

df = pd.read_hdf('dataframe.h5', key = 'table')

print(df)

# print(df.groupby(level = [driver]).apply(len).reindex(index = (df)))

# for driver, trip in df.groupby(level = 1):
#    print(driver + "_" + trip)
    