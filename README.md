# Driver Telematics Analysis - Team PIKKI

The code reached an AUC of 0.8850 on the public leaderboard on kaggle
for the [Driver Telematic Analysis
](http://www.kaggle.com/c/axa-driver-telematics-analysis/).

The code is structured as follows:

1. Step_1_Create_DataFrame.py:

In this step, the entire kaggle data set is read into data frames
and saved in HDF5 files. This step only has to be done once, and
reduces the time required for Step 2 and 3.

2. Step_2_Extract_Features.py:

Here, features for each trip are extracted, put in a data frame and
saved in HDF5 format.

3. Step_3_Classify.py:

In this step, a supervised learning approach (Random Forest
Classifier) is used. For this, a classifier is trained for each driver as
positive set and 200 random trips from other drivers are used as negative
training set.