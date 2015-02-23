import pandas as pd
import numpy as np
from os import path, listdir
from sklearn import svm
from sklearn import linear_model
import scipy as sp
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier


def create_submission_file(df):
    """
    Create a submission file for kaggle
    """

    # Find file number for new file
    file_num = 0
    while path.isfile('submission-{}.csv'.format(file_num)):
        file_num += 1

    # Write final submission
    df.to_csv('submission-{}.csv'.format(file_num), index = False)


def calc_prob(df_features):


    #model = BaggingClassifier(base_estimator = RandomForestClassifier())
    model = BaggingClassifier(base_estimator = ExtraTreesClassifier())
    # model = BaggingClassifier(base_estimator = svm.SVC(gamma=2, C=1))
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    # model = BaggingClassifier(base_estimator = AdaBoostClassifier())E
    # model = RandomForestClassifier()
    model.fit(df_features.ix[:, 4:], df_features.Driver)

    df_submission = pd.DataFrame()
    df_submission['driver_trip'] = create_first_column(df_features)
    # df_submission.reset_index(inplace = True)

    probs_array = model.predict_proba(df_features.ix[:, 4:]) # Return array with the probability for every driver
    probs_df = pd.DataFrame(probs_array)

    #drivers = listdir(r"C:\Users\User\PycharmProjects\awesome-kagg-ml\drivers")
    drivers = df_features.Driver.unique()
    drivers.sort()
    drivers_dict = dict(zip(drivers, range(len(drivers))))

    probs_list = []
    for index, row in df_features.iterrows():
        driver = row['Driver']
        driver_pos_in_array = drivers_dict[str(driver).rstrip()]

        probs_list.append(probs_df.ix[index, driver_pos_in_array])

    df_submission['prob'] = probs_list

    return df_submission


def create_first_column(df):
    return df.ix[:, 2].apply(str) + "_" + df.ix[:, 3].apply(str)

### Features (should work on trip level and return one single number)
## TODO: Should be placed in another file


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def main():
    # chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks_big"
    # chunk_path = r"/home/pold/Documents/Radboud/kaggle/chunks"


    # features_path = r"/home/pold/Documents/Radboud/kaggle/features"
    features_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\features"

    features_files = listdir(features_path)

    features_df_list = [pd.read_hdf(path.join(features_path, f), key = 'table') for f in features_files]


    feature_df = pd.concat(features_df_list)

    # feature_df = pd.read_hdf(r"C:\Users\User\PycharmProjects\awesome-kagg-ml\Code\feature_df.h5", key = 'table')


    #feature_df = pd.read_hdf(r"/home/pold/Documents/Radboud/kaggle/Code/feature_df.h5", key = 'table')
    feature_df.reset_index(inplace = True)
    df_list = []


    nr_of_drivers_for_comparison = 5

    # Split drivers in parts
    # TODO: Insert random drivers from the entire data set (maybe)
    chunked_drivers = chunks(feature_df.index, 200 * nr_of_drivers_for_comparison)

    for part in chunked_drivers:

        part_df = feature_df.ix[part]

        part_df.reset_index(inplace = True)

        submission_df = calc_prob(part_df)
        df_list.append(submission_df)

    submission_df = pd.concat(df_list)
    create_submission_file(submission_df)


if __name__ == "__main__":
    main()