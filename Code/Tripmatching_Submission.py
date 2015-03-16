import pandas as pd
import numpy as np
from os import path, listdir
import re
import h5py


def create_first_column(df):
    """
    Create first column for the submission csv, e.g.
    driver_trip
    1_1
    1_2
    """
    return df.Driver.apply(str) + "_" + df.Trip.apply(str)

def create_submission_file(df):
    """
    Create a submission file for kaggle from a data frame
    """

    # Find file number for new file
    file_num = 0
    while path.isfile('submission-matched-{}.csv'.format(file_num)):
        file_num += 1

    # Write final submission
    df.to_csv('submission-matched-{}.csv'.format(file_num), index = False)

matched = listdir("/scratch/vstrobel/matched/")

df_list = []

for f in matched:
    d = re.findall('(\d+)', f)[0]
    n = 'dataset_' + d

    h5f = h5py.File("/scratch/vstrobel/matched/" + f, 'r')
    weights_matrix = h5f[n][:]
    h5f.close()

    weights = np.amax(weights_matrix[1:, 1:], axis = 1)

    df = pd.DataFrame(columns = ['Driver', 'Trip', 'prob'])
    df['Driver'] = np.repeat(int(d), 200)
    df['Trip'] = np.arange(1,201)
    df['prob'] = weights

    df_list.append(df)


total = pd.concat(df_list)

total.to_csv('weights.csv', index = False)


# total['driver_trip'] = create_first_column(total)

# submission = total[['driver_trip', 'prob']]

# create_submission_file(submission)















