import pandas as pd
import numpy as np
from os import path, listdir
import matplotlib.pyplot as plt
import time
import AUC
from Stack import *
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor

def crossvalidation(df_features_driver, df_features_other, nfold, model):

    df_auc = []

    for n in range(nfold):
        len_fold = int(len(df_features_driver)/nfold)
        ind_train = np.append(np.arange(0,int(n)*len_fold,1),
                              np.arange((int(n)+1)*len_fold,len(df_features_driver),1))
        ind_test = np.arange(int(n)*len_fold,(int(n)+1)*len_fold,1)
        df_train = df_features_driver.iloc[ind_train].append(df_features_other.iloc[ind_train])
        df_train.reset_index(inplace = True)
        df_train.Driver = df_train.Driver.astype(int)    
    
        df_test = df_features_driver.iloc[ind_test].append(df_features_other.iloc[ind_test])
        df_test.reset_index(inplace = True)
        df_test.Driver = df_test.Driver.astype(int)        

        feature_columns_train= df_train.iloc[:, 4:]
        feature_columns_test= df_test.iloc[:, 4:]

        # Train the classifier
        model.fit(feature_columns_train, df_train.Driver)        
        
        probs_array = model.predict_proba(feature_columns_test) # Return array with the probability for every driver
        probs_df = pd.DataFrame(probs_array)

        probs_list = np.array(['1', probs_df.ix[0, 1]])
        for x in range(1, len_fold):
            # Column 1 should contain the driver of interest
            probs_list = np.vstack((probs_list, ['1', probs_df.ix[x, 1]]))
        for x in range(len_fold,2*len_fold):
            # Column 1 should contain the driver of interest
            probs_list = np.vstack((probs_list, ['0', probs_df.ix[x, 1]]))
    
        df_auc.append(AUC.AUC(probs_list))  
    
    return np.mean(df_auc)  
    
    
def main():

    features_path = path.join('..', 'features_small')
    features_files = listdir(features_path)

    # Get data frame that contains each trip with its features
    features_df_list = [pd.read_hdf(path.join(features_path, f), key = 'table') for f in features_files]
    feature_df = pd.concat(features_df_list)
    feature_df.reset_index(inplace = True)
    df_list = []
    nfold = 10 # either 2, 4, 5, 10, or 20
    t0 = time.time()

    n_trees = 10
    # Generate a list of base (level 0) classifiers.
    clfs = [RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=n_trees, n_jobs=-1, criterion='entropy'),
        #GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=n_trees)
        ]

    n_folds = 5

    # Generate k stratified folds of the training data.
    skf = list(cross_validation.StratifiedKFold(y_train, n_folds))


    model0 = Stacking(LogisticRegression, clfs, skf, stackingc=False, proba=True)
    model1 = AdaBoostClassifier()
    model2 = AdaBoostClassifier(n_estimators=200)
    model3 = AdaBoostClassifier(n_estimators=200, base_estimator = RandomForestClassifier(100))
    model4 = ExtraTreesClassifier(n_estimators=2000, criterion='entropy')
    model5 = ExtraTreesClassifier(n_estimators=10000)


    models = [model1
         , model2
        , model3
         , model4
         , model5
    #     , model6
    ]

    for model in models:

        for i, (_, driver_df) in enumerate(feature_df.groupby('Driver')):

            indeces = np.append(np.arange(0,int(i)*200,1),np.arange((int(i)+1)*200,len(feature_df),1))
            # Get 400 other trips
            other_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]

            others = feature_df.iloc[other_trips]

            others.Driver = int(0)

            crossvalidation_df = crossvalidation(driver_df, others, nfold, model)
            df_list.append(crossvalidation_df)
           # if i % 100 == 0:
           #     print(i, ': ', time.time() - t0)


        #plt.hist(df_list, bins=100, normed=1)
        #plt.show()
        print(np.mean(df_list))


if __name__ == "__main__":
    main()