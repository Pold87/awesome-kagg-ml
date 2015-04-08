
#%%writefile otto_Pikki.py
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
#from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2, f_classif
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.lda import LDA
#from sklearn.neural_network import BernoulliRBM
#from sklearn.svm import LinearSVC
from sklearn import cross_validation
#from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                           #       LassoLarsCV)




from matplotlib import pyplot as plt

def sigmoid(x):
    print('XXXX')
    y = 1/(1+np.exp(x))
    return y
#%matplotlib inline
import time

def load_data(train=True):
    """loads Data.
    train == True means training data which has y values
    train == False means testing (=pleasePredict) data, which only has X values
    """
#test set
    start = time.clock()

    if train==False:
        dp = '/home/thomas/Desktop/OttoChallenge/test.csv'
        df = pd.read_csv(dp)
        X = df.values.astype(np.float32)[:,1:]
        return X
#train set
    else:
        dp = '/home/thomas/Desktop/OttoChallenge/train.csv'
        df = pd.read_csv(dp)
        X = df[df.columns[:-1]].values.astype(np.float32)[:,1:]
        y = df.target
        y =y.apply(lambda X: int(X[-1])).values
        y = y.astype(np.int32)
        X, y = shuffle(X, y)
        #print(X.shape,y.shape)
        end = time.clock()
        print(end-start)

        return X,y


def generate_features(X, shuffle_data = False, which_slice = 0,how_many_slices = 1,dummies = False):
    """generates features from np.array X.
    most of them aren't that great but give a little boost in performance. Last 93 features are interesting,
    computed from a correlation matrix.
    
    shuffle Data shuffles the data,
    how many slices specifies by what number the data should be divided
    which slice then states which part of the data should be used (mind the 0)
    dummes keyword doesn't work yet.
    """
    start = time.clock()
 
    
    df_X = pd.DataFrame(X)
    if dummies ==True:
        print('dummies')
        
    df_tmp = df_X.cumsum(axis=1)
    
    df_tmp_2 = df_X.cumprod(axis=1)
    df_tmp_5 = df_X.cummax(axis=1)

    df_tmp_6 = df_X.cummin(axis=1)
    df_tmp_3 = df_X**2
    df_tmp_4 =     df_X.T.diff(2).T.dropna(axis=1,how='any')
#df_X**3
    df_tmp_7 = df_X.T.diff(1).T.dropna(axis=1,how='any')
    df_tmp_8 = df_X**3
    df_tmp_9 = df_X**5
    df_tmp_10 = df_X - df_X.apply(np.mean,axis=0)#will this work?
    df_test = pd.stats.moments.rolling_mean(df_X.T,window = 19,axis=1)
    df_test = df_test.T.dropna(axis=1,how='any')

    df_X['nonzero_sum'] = df_X.apply(np.count_nonzero,axis=1)
   # df_X['binc'] = df_X.apply(np.unique,axis=1)
    corr = df_X.corr()
    df_tmp_11 = df_X.dot(corr)

    
    
    
    df_X['mean'] = df_X.apply(np.mean,axis=1,raw=True)
    df_X['std'] = df_X.apply(np.std,axis=1,raw=True)
    df_X['median'] = df_X.apply(np.median,axis=1,raw=True)
    df_X['amax'] = df_X.apply(np.amax,axis=1,raw=True)
    df_X['amin'] = df_X.apply(np.amin,axis=1,raw=True)


    df_X['ptp'] = df_X.apply(np.ptp,axis=1)
    
    
    df_X = pd.concat([df_X,
                      df_tmp,
                      df_tmp_2,
                      #df_tmp_3,
                      df_tmp_4,
                      df_tmp_5,
                      df_tmp_6,
                      #df_tmp_7,
                      #df_tmp_8,
                      #df_tmp_9,
              df_tmp_11,
                      df_test
                      ],axis=1)
        
    X = df_X.values.astype(np.float32)

    if shuffle_data == True:
        z = shuffle(X.T)
        X = z.T
    print('shape')
    print(X.shape)
    
    slice_length = round(X.shape[1]/how_many_slices)
    print(slice_length, how_many_slices)
    X = X[:,which_slice*slice_length:(which_slice+1)*slice_length]
    
    print('shape new')
    print(X.shape)
    end = time.clock()

    return X



from sklearn.metrics import log_loss




def get_score(clf, features = True,which_slice = 0,how_many_slices = 1,shuffle_data = False):
    """computes a cross validation score
    
    takes all keywords from generate_features"""
    
    start = time.clock()
    X,y = load_data()
    if features == True:
        X = generate_features(X,which_slice=which_slice,how_many_slices=how_many_slices ,shuffle_data=shuffle_data)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
    #print(y_train)
    clf.fit(X_train,y_train)
    outcome = clf.predict_proba(X_test)
    end = time.clock()
    print('SCORE: ' , log_loss(y_test,outcome), ' time: ',end-start)
    
def get_score2(clf,features = True,which_slice = 1,how_many_slices = 5,shuffle_data = False):
    X,y = load_data()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)#, random_state=0)
    clf.fit(X_train,y_train)
    outcome = clf.predict_proba(X_test)
    print(log_loss(y_test,outcome))






def make_guess_csv(clf,features =True,name = 'test',which_slice = 0,how_many_slices = 1,shuffle_data = False):
    """generates a csv file containing the output from one classifier.
    index starts at 0, so no submittable file is created (was necessary in an older version of combine_results, 
    not sure if it still is)
    """
    start = time.clock()
    X_test,y_test = load_data()
    X = load_data(train=False)
    if features == True:
        X = generate_features(X,which_slice=which_slice ,how_many_slices=how_many_slices ,shuffle_data=shuffle_data)
        X_test = generate_features(X_test,which_slice=which_slice,how_many_slices=how_many_slices ,shuffle_data=shuffle_data )

    
    clf.fit(X_test,y_test)
    output = clf.predict_proba(X)
    
    
    cols = ['Class_1',
      'Class_2',
      'Class_3',
      'Class_4',
      'Class_5',
      'Class_6',
      'Class_7',
      'Class_8',
      'Class_9']
    df_write = pd.DataFrame(output)
    shuf = '_noshuf_'
    if shuffle_data==True:
        shuf = '_shufled_'
    slices = 'num_slice_' + str(how_many_slices)
    which_clf = clf
    num_slice = 'slice_num_' + str(which_slice)
    clf = str(clf)
    
    file_name = 'Pikki' + name  +num_slice + slices + shuf + '.csv'
    print('written to ' + file_name)
    df_write.to_csv(file_name,header = cols,index_label = ['id'])
    end = time.clock()
    print(end-start)
    identifier = name  +num_slice + slices + shuf
    return identifier
    


def load_all_dfs(clf_list = ['test_small','rt_small','test2_small']):
    """loads all the csv files that end with specification in clf_list
    in multiindex DataFrame. first row is specification, second is index
    
    helper for combine_results and plot_bias
    """
    
    start = time.clock()
    print('loading data')
    first_clf = clf_list[0]
    df = pd.read_csv('Pikki'+first_clf+'.csv')
    df['df'] = first_clf

    df = df.set_index(['id','df'])

    for clf in clf_list[1:]:
        file_name = 'Pikki' + clf + '.csv'
        df_tmp = pd.read_csv(file_name)
        df_tmp['df'] = clf

        df_tmp = df_tmp.set_index(['id','df'])

        df = pd.concat([df,df_tmp])

        
    df['std'] = df.apply(np.std,axis=1,raw = True)
    end = time.clock()
    print(end-start)
    return df#.swaplevel(0,1)




def combine_results(voting = 'hard',clf_list = ['test_small','rt_small','test2_small']):
    """combines two or more classifier outputs. 
    by now only uses hard voting by standard deviation 
    (more std means higher values which ~ means higher confidence)
    other measures welcome
    
    creates both a submittable file ("combined ...")
    and a file that can be used for combination with other classifiers
    """
    
    start = time.clock()
    df = load_all_dfs(clf_list)

    print('combining the data and voting ', voting)

    if voting == 'hard':
        print('voting')

        label_tupel_list = list(df.groupby(level=['id'])['std'].idxmax())#idmax 
        num_samples = len(label_tupel_list)
        index = [label_tupel_list[i][0] for i in range(num_samples)]
        df.index
        time_need = []
        t2 = 0

        print("doing god's work")
        df_new = df.ix[index]
        df_new = df.ix[label_tupel_list]
    end = time.clock()
    print('done', end-start)
    #return df_new
   
    
    cols = ['Class_1',
      'Class_2',
      'Class_3',
      'Class_4',
      'Class_5',
      'Class_6',
      'Class_7',
      'Class_8',
      'Class_9']
    df_new2 = df_new.reset_index()
    del df_new2['std']
    del df_new2['id']
    del df_new2['df']

    print('zero')
    try:
     print('first')
     clf_names = 'with_'
     print('second')
     for i in range(len(clf_list)):
         print(clf_list[i])
         clf_names = clf_names + '_' +  clf_list[i]
            
     df_new2.to_csv('Pikki'+clf_names+ '.csv',header = cols,index_label = ['id'])
       
     df_new2.index +=1

     print('written to')
     print('Pikki'+clf_names+ '.csv')
     
     df_new2.to_csv('combined_Pikki'+clf_names+ '.csv',header = cols,index_label = ['id'])
    except:
        df_new2.to_csv('combined_Pikki.csv',header = cols,index_label = ['id'])
    return df_new    



def plot_bias(clf_list = ['test_small','rt_small','test2_small'],return_df = False,XKCD = False):
    """plots some differences between two output files.
    right now plots for every class and every classifier
    
    mean
    max
    std
    diff(clfx-clfy)
    
    plt.xkcd() is optional.
    """
    if XKCD = True:
        plt.xkcd()
    print('damn')
    df = load_all_dfs(clf_list)
    df = df.swaplevel(0,1)
    del df['std']
    df.hist()
    plt.figure()

    for clf in clf_list:
        df.ix[clf].mean().plot(label = clf,figsize=(16, 4))
    plt.legend(loc='upper right')
    plt.title('mean')
    plt.figure()
    
   # c = df.columns
    for clf in clf_list:
        #df[c[1:]].ix[clf].max().plot(label = clf,figsize=(16, 4))
        df.ix[clf].max().plot(label = clf,figsize=(16, 4))
    plt.legend(loc='upper right')
    plt.title('max')
    
    plt.figure()
    for clf in clf_list:
        df.ix[clf].std().plot(label = clf,figsize=(16, 4))

        
    plt.legend(loc='upper right')
    plt.title('std')
    plt.figure()
    used_list = []
    for clf in clf_list:
        for clf2 in clf_list:
            if (clf != clf2) and ({clf,clf2} not in used_list):
                diff = ((df.ix[clf] - df.ix[clf2])**2)**(1/2)
                diff.mean().plot(label = clf+' - ' +clf2,figsize=(16, 4))
                used_list.append({clf,clf2})
                
                
                
                
                
    plt.legend(loc='upper right')
    plt.title('difference')
    print('damnover')
    if return_df == True:
     return df
    
def example():
    """example on how to run this"""
    #generate a classifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    clf = RandomForestClassifier(n_estimators=4,n_jobs=-1)
    clf2 = GradientBoostingClassifier(n_estimators = 1)
    #make guess csv
    n1 = make_guess_csv(clf,name='example')
    n2 = make_guess_csv(clf2,name='example_2')
    #combine results
    print(n1,n2)
    combine_results(clf_list=[n1,n2])
    #plot the stuff
    plot_bias([n1,n2],XKCD = True)
    
#noclue if this is necessary. If somebody has better information, please let me know
def main():
    example()
if __name__ == "__main__":
    main()
    