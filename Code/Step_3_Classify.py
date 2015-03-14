from __future__ import division
import pandas as pd
import numpy as np
from os import path, listdir
import Helpers
from sklearn import svm
from sklearn import linear_model
import scipy as sp
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor
import multiprocessing as mp

weights = np.array([])


def create_submission_file(df):
    """
    Create a submission file for kaggle from a data frame
    """

    # Find file number for new file
    file_num = 0
    while path.isfile('submission-{}.csv'.format(file_num)):
        file_num += 1

    # Write final submission
    df.to_csv('submission-{}.csv'.format(file_num), index = False)


def classify(f, df_list, weights):

    feature_df = pd.read_hdf("/scratch/vstrobel/features/" + f, key = 'table')
    feature_df.reset_index(inplace = True)
    feature_df = feature_df.sort(['Driver', 'Trip'])

    calculated = []


    print(weights)

    for i, (d, driver_df) in enumerate(feature_df.groupby('Driver')):

        weights_mask = weights['Driver'] == int(d)
        weights_driver = weights[weights_mask].prob

        print(weights_driver)
        
        amount_others = 200
        weights_others = weights_driver.mean() * np.ones(amount_others)
    
        indeces = np.append(np.arange(i * 200), np.arange((i+1) * 200, len(feature_df)))
        other_trips = indeces[np.random.randint(0, len(indeces) - 1, amount_others)]
        others = feature_df.iloc[other_trips]
        others.Driver = np.repeat(int(0), amount_others)
    
        submission_df = calc_prob(driver_df, others, np.append(weights_driver, weights_others))
        calculated.append(submission_df)

    df_list.append(pd.concat(calculated))

def calc_prob(df_features_driver, df_features_other, weights):

    df_train = df_features_driver.append(df_features_other)
    df_train.reset_index(inplace = True)
    df_train.Driver = df_train.Driver.astype(int)

    model = RandomForestClassifier(n_estimators = 10000, min_samples_leaf=2, sample_weight = weights)
    feature_columns = df_train.iloc[:, 4:]

    # Train the classifier
    model.fit(feature_columns, df_train.Driver, sample_weight=weights)
    df_submission = pd.DataFrame()

    df_submission['driver_trip'] = create_first_column(df_features_driver)

    probs_array = model.predict_proba(feature_columns[:200]) # Return array with the probability for every driver
    probs_df = pd.DataFrame(probs_array)

    df_submission['prob'] = np.array(probs_df.iloc[:, 1])

    return df_submission

def create_first_column(df):
    """
    Create first column for the submission csv, e.g.
    driver_trip
    1_1
    1_2
    """
    return df.Driver.apply(str) + "_" + df.Trip.apply(str)


def main():

    features_path = "/scratch/vstrobel/features_angles_32"
    features_files = sorted(listdir(features_path))

    matched_probs = pd.read_csv("weights.csv")
    matched_probs = matched_probs.sort(['Driver', 'Trip'])

    # Get data frame that contains each trip with its features
    
    manager = mp.Manager()
    df_list = manager.list()

    jobs = []

    for f in features_files:
        p = mp.Process(target = classify, args = (f, df_list, matched_probs, ))
        jobs.append(p)
        p.start()
        
    [job.join() for job in jobs]

    final_list = []

    for l in df_list:
        final_list.append(l)

    submission_df = pd.concat(final_list)
    create_submission_file(submission_df)

if __name__ == "__main__":
    main()
