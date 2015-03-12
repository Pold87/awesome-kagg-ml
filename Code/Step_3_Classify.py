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
# from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor


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


def calc_prob(df_features_driver, df_features_other, weights):

    df_train = df_features_driver.append(df_features_other)
    df_train.reset_index(inplace = True)
    df_train.Driver = df_train.Driver.astype(int)

    model = RandomForestClassifier(n_estimators = 1000, min_samples_leaf=2)

    # So far, the best result was achieved by using a RandomForestClassifier with Bagging
    # model = BaggingClassifier(base_estimator = ExtraTreesClassifier())
    # model = BaggingClassifier(base_estimator = svm.SVC(gamma=2, C=1))
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    # model = BaggingClassifier(base_estimator = AdaBoostClassifier())

    # model = BaggingClassifier(base_estimator = [RandomForestClassifier(), linear_model.LogisticRegression()])
    # model = EnsembleClassifier([BaggingClassifier(base_estimator = RandomForestClassifier()),
    #                             GradientBoostingClassifier])
    # model = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.05, random_state=0, subsample = 0.85)
    # model = GradientBoostingRegressor(n_estimators = 1000)
    # model = RandomForestClassifier(500)

    feature_columns = df_train.iloc[:, 4:]

    # Train the classifier
    model.fit(feature_columns, df_train.Driver, sample_weight=weights)
    df_submission = pd.DataFrame()

    df_submission['driver_trip'] = create_first_column(df_features_driver)

    probs_array = model.predict_proba(feature_columns[:200]) # Return array with the probability for every driver
    probs_df = pd.DataFrame(probs_array)

    # wrong_probs_array = model.predict_proba(feature_columns[200:]) # Return array with the probability for every driver
    # wrong_probs_df = pd.DataFrame(wrong_probs_array)
    # print(wrong_probs_df)

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

    features_path_1 = path.join('..', 'features')
    features_files_1 = listdir(features_path_1)
    
    #features_path_2 = path.join('..', 'features_2')
    #features_files_2 = listdir(features_path_2)

    # Get data frame that contains each trip with its features
    features_df_list_1 = [pd.read_hdf(path.join(features_path_1, f), key = 'table') for f in features_files_1]
    feature_df_1 = pd.concat(features_df_list_1)
    
    #features_df_list_2 = [pd.read_hdf(path.join(features_path_2, f), key = 'table') for f in features_files_2]
    #feature_df_2 = pd.concat(features_df_list_2)  
    #feature_df_2x = feature_df_2[['Driver', 'Trip', 'mean_speed_times_acceleration', 'pauses_length_mean']]    
    
    # feature_df = pd.merge(feature_df_1, feature_df_2x, on=['Driver', 'Trip'], sort = False)
    
    feature_df = feature_df_1    
    
    feature_df.reset_index(inplace = True)
    df_list = []

    stacks = 1

    for i, (_, driver_df) in enumerate(feature_df.groupby('Driver')):

        weights_driver = np.ones(200)


        for s in reversed(range(stacks)):

            # amount_others = (s + 1) * 25
            amount_others = 200
            weights_others = weights_driver.mean() * np.ones(amount_others)

            indeces = np.append(np.arange(i * 200), np.arange((i+1) * 200, len(feature_df)))
            other_trips = indeces[np.random.randint(0, len(indeces) - 1, amount_others)]
            others = feature_df.iloc[other_trips]
            others.Driver = int(0)
            # others['weights'] = weights_others
            # driver_df['weights'] = weights_driver

            submission_df = calc_prob(driver_df, others, np.append(weights_driver, weights_others))
            # weights_driver = submission_df.prob
            # print(weights_driver)


            sorted_df = submission_df.sort(['prob'])
            step = 1 / 200
            new_probs = np.arange(0, 1, step)

            sorted_df['prob'] = new_probs

        df_list.append(sorted_df)

    submission_df = pd.concat(df_list)
    weights = submission_df.iloc[:, 1]


    create_submission_file(submission_df)


if __name__ == "__main__":
    main()