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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor



def minusOneToZero(x):
    if x > 0.5:
        return 1
    else:
        return 0

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers
        self.predictions_ = list()

    def fit(self, x, y):
        for classifier in self.classifiers:
            classifier.fit(x, y)

    def predict_proba(self, x):
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(x))
            m = np.mean(self.predictions_, axis=0)
        return m

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


def calc_prob(df_features_driver, df_features_other, df_features_test_trips):

    df_train = df_features_driver.append(df_features_other)
    df_train.reset_index(inplace = True)
    df_train.Driver = df_train.Driver.astype(int)

    df_features_test_trips.reset_index(inplace = True)
    feature_test_columns = df_features_test_trips.iloc[:, 4:]

    # So far, the best result was achieved by using a RandomForestClassifier with Bagging
    # model = BaggingClassifier(base_estimator = ExtraTreesClassifier())
    # model = BaggingClassifier(base_estimator = svm.SVC(gamma=2, C=1))
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    model = RandomForestClassifier(100, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 2)
    # model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
    # model = BaggingClassifier(base_estimator = AdaBoostClassifier())
    # model = RandomForestClassifier(1000, n_jobs=-1)
    # model = BaggingClassifier(base_estimator = [RandomForestClassifier(), linear_model.LogisticRegression()])
    # model = EnsembleClassifier([BaggingClassifier(base_estimator = RandomForestClassifier()),
    #                             GradientBoostingClassifier])
    # model = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.05, random_state=0, subsample = 0.85)
    # model = GradientBoostingRegressor(n_estimators = 1000)
    # model = ExtraTreesClassifier(500, criterion='entropy')

    feature_columns = df_train.iloc[:, 4:]

    # Train the classifier
    model.fit(feature_columns, df_train.Driver, sample_weight= np.append(np.ones(200), 10 * np.ones(200)))
    # df_submission = pd.DataFrame()

    # df_submission['driver_trip'] = create_first_column(df_features_driver)

    hopefully_rejected = model.predict_proba(feature_test_columns) # Return array with the probability for every driver
    # probs_df = pd.DataFrame(probs_array)

    hopefully_rejected_df = pd.DataFrame(hopefully_rejected)

    score = hopefully_rejected_df.loc[:, 0].mean()

    return score

def create_first_column(df):
    """
    Create first column for the submission csv, e.g.
    driver_trip
    1_1
    1_2
    """
    return df.Driver.apply(str) + "_" + df.Trip.apply(str)


def main():

    features_path_1 = path.join('..', 'features_small')
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


    # model1 = RandomForestClassifier(n_estimators=10)
    # model1 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 1)
    # model2 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 2)
    # model3 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 3)
    # model4 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 4)
    model1 = GradientBoostingClassifier(500)
    model2 = GradientBoostingClassifier(1000)
    model3 = GradientBoostingClassifier(1000)
    model4 = GradientBoostingClassifier(1000)
    model5 = GradientBoostingClassifier(1000)
    # model1 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, min_samples_split = 1)
    # model2 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, max_leaf_nodes = 2)
    # model3 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, max_leaf_nodes = 2)
    # model4 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, max_leaf_nodes = 3)
    # model5 = RandomForestClassifier(1000, n_jobs=-1, min_samples_leaf = 2, max_leaf_nodes = 4)
    # model2 = NuSVC(nu = 0.2)
    # model3 = NuSVC(nu = 0.3)
    # model4 = NuSVC(nu = 0.4)
    # model5 = NuSVC(nu = 0.5)
    # model1 = OneClassSVM(kernel = 'sigmoid')
    #model2 = RandomForestClassifier(n_estimators=200, max_features='log2', criterion='entropy')
    #model3 = RandomForestClassifier(n_estimators=500, max_features='log2', criterion='entropy')
    # model4 = RandomForestClassifier(n_estimators=200, bootstrap=False)
    # model5 = RandomForestClassifier(n_estimators=200, oob_score=False)
    # model6 = RandomForestClassifier(n_estimators=200, oob_score=True)
    # model7 = RandomForestClassifier(n_estimators=200, random_state=0)
    # model8 = NuSVR()
    # model9 = NuSVR(C = 0.5)
    # model10 = NuSVR(kernel = 'sigmoid')
    # model11 = NuSVR(nu = 0.7)


    models = [model1
         , model2
         , model3
         , model4
         , model5
         # , model6
         # , model7
         # , model8
         # , model9
         # , model10
         # , model11
    ]

    stacks = 10

    for model in models:


        for i, (_, driver_df) in enumerate(feature_df.groupby('Driver')):

            indeces = np.append(np.arange(i * 200), np.arange((i+1) * 200, len(feature_df)))
            other_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]

            test_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]
            test = feature_df.iloc[test_trips]

            others = feature_df.iloc[other_trips]
            others.Driver = int(0)

            submission_df = calc_prob(driver_df, others, test)
            df_list.append(submission_df)

        final_score = np.array(df_list).mean()
        print(final_score)
        # create_submission_file(submission_df)


if __name__ == "__main__":
    main()