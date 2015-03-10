import pandas as pd
import numpy as np
from os import path, listdir
import matplotlib.pyplot as plt
import time
import AUC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import NuSVR, OneClassSVM, NuSVC



def minusOneToZero(x):
    if x > 0.5:
        return 1
    else:
        return 0

def crossvalidation(df_features_driver, df_features_other, nfold, model):

    df_auc = []

    for n in range(nfold):

        classification = True

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

        if classification:
            probs_df = pd.DataFrame()
            probs_df['other'] = 0
            probs_df['driver'] = model.predict(feature_columns_test)
            probs_df['driver'] = probs_df['driver'].apply(minusOneToZero)

            # print(probs_df['driver'])

        else:
            probs_array = model.predict_proba(feature_columns_test) # Return array with the probability for every driver
            probs_df = pd.DataFrame(probs_array)
            print(probs_df.iloc[:, 1])

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

    # model1 = RandomForestClassifier(n_estimators=10)
    model1 = NuSVC(nu = 0.1)
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
         # , model2
         # , model3
         #  , model4
         #  , model5
         # , model6
         # , model7
         # , model8
         # , model9
         # , model10
         # , model11
    ]

    for model in models:

        for i, (_, driver_df) in enumerate(feature_df.groupby('Driver')):

            indeces = np.append(np.arange(0,int(i)*200,1),np.arange((int(i)+1)*200,len(feature_df),1))
            # Get 400 other trips
            other_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]

            others = feature_df.iloc[other_trips]


            others.Driver = int(0)

            driver_df.Driver = int(1)

            crossvalidation_df = crossvalidation(driver_df, others, nfold, model)
            df_list.append(crossvalidation_df)
           # if i % 100 == 0:
           #     print(i, ': ', time.time() - t0)


        #plt.hist(df_list, bins=100, normed=1)
        #plt.show()
        print(np.mean(df_list))


if __name__ == "__main__":
    main()