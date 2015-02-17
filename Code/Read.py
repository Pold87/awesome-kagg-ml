import pandas as pd
import numpy as np
from os import path, listdir

def listdir_fullpath(d):
    return [path.join(d, f) for f in listdir(d)]


chunks = listdir_fullpath("/home/pold/Documents/Radboud/kaggle/chunks")

chunks_small = [chunks[2]]


chunk_list = []

for chunk in chunks_small:

    df = pd.read_hdf(chunk, key = 'table')
    chunk_list.append(df)

df_all = pd.concat(chunk_list)

print(df_all)

# Pickle the dataframe
# df_all.to_hdf("Total.h5",'table')
    