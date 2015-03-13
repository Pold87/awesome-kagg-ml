import pandas as pd
import numpy as np
from os import path, listdir
import matplotlib.pyplot as plt
import time
import AUC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import NuSVR, OneClassSVM, NuSVC


def cross_classify(df_features_driver, df_features_other, df_features_test_trips,  nfold, model):

    df_submission = pd.DataFrame()
    
    df_submission['driver_trip'] = create_first_column(df_features_driver)

    all_probs = []
    all_wrong_probs = []

    for n in range(nfold):
        
        len_fold = int(len(df_features_driver)/nfold)
        len_wrong_fold = int(len(df_features_test_trips)/nfold)
        ind_train = np.append(np.arange(0,int(n)*len_fold,1),
                              np.arange((int(n)+1)*len_fold,len(df_features_driver),1))
        ind_test = np.arange(int(n)*len_fold,(int(n)+1)*len_fold,1)

        ind_wrong = np.arange(int(n)*len_wrong_fold,(int(n)+1)*len_fold,1)
        
        df_train = df_features_driver.append(df_features_other)
        df_train.reset_index(inplace = True)
        df_train.Driver = df_train.Driver.astype(int)
        
        df_train = df_features_driver.iloc[ind_train].append(df_features_other.iloc[ind_train])
        df_train.reset_index(inplace = True)
        df_train.Driver = df_train.Driver.astype(int)    
    
        df_test = df_features_driver.iloc[ind_test]
        df_test.reset_index(inplace = True)
        df_test.Driver = df_test.Driver.astype(int)

        df_wrong = df_features_test_trips.iloc[ind_wrong]
        df_wrong.reset_index(inplace = True)
        df_wrong.Driver = df_wrong.Driver.astype(int)
    
        feature_columns_train= df_train.iloc[:, 4:]
        feature_columns_test= df_test.iloc[:, 4:]
        feature_columns_wrong = df_wrong.iloc[:, 4:]

        # Train the classifier
        model.fit(feature_columns_train, df_train.Driver)

        probs_array = model.predict_proba(feature_columns_test[:])

        wrong_probs_array = model.predict_proba(feature_columns_wrong[:])

        all_probs = np.append(all_probs, np.array(probs_array[:, 1]))
        all_wrong_probs = np.append(all_wrong_probs, wrong_probs_array[:, 0])


    # hopefully_rejected = model.predict_proba(feature_test_columns) # Return array with the probability for every driver
    probs_df = pd.DataFrame(probs_array)
    #
    # hopefully_rejected_df = pd.DataFrame(hopefully_rejected)

    # score = hopefully_rejected_df.loc[:, 0].mean()

    score1 = all_wrong_probs.mean()
    score2 = all_probs.mean()


    return score1, score2



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

    # Get data frame that contains each trip with its features
    features_df_list_1 = [pd.read_hdf(path.join(features_path_1, f), key = 'table') for f in features_files_1]
    feature_df_1 = pd.concat(features_df_list_1)
    
    feature_df = feature_df_1    
    
    feature_df.reset_index(inplace = True)
    df_list1 = []
    df_list2 = []

    model1 = GradientBoostingClassifier(n_estimators=5000, min_samples_leaf=2)
    model4 = ExtraTreesClassifier()
    model5 = RandomForestClassifier()

    models = [model1
         , model4
         , model5
    ]


    # k-fold
    nfold = 20

    for model in models:

        for i, (_, driver_df) in enumerate(feature_df.groupby('Driver')):
            indeces = np.append(np.arange(i * 200), np.arange((i+1) * 200, len(feature_df)))

            other_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]
            others = feature_df.iloc[other_trips]
            others.Driver = int(0)

            test_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]
            test = feature_df.iloc[test_trips]

            score1, score2 = cross_classify(driver_df, others, test, nfold, model)
            df_list1.append(score1)
            df_list2.append(score2)

        print(np.array(df_list1).mean())
        print(np.array(df_list2).mean())


if __name__ == "__main__":
    main()