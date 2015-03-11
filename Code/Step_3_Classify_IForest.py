
import pandas as pd
import numpy as np
from os import path, listdir
import iForest
import sys

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


def calc_prob(df_features_driver, df_features_other):

	df_train = df_features_driver.append(df_features_other)
	df_train.reset_index(inplace = True)
	df_train.Driver = df_train.Driver.astype(int)

	# So far, the best result was achieved by using a RandomForestClassifier with Bagging
	# model = BaggingClassifier(base_estimator = ExtraTreesClassifier())
	# model = BaggingClassifier(base_estimator = svm.SVC(gamma=2, C=1))
	# model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
	# model = BaggingClassifier(base_estimator = linear_model.LogisticRegression())
	# model = BaggingClassifier(base_estimator = AdaBoostClassifier())
	model = RandomForestClassifier(150)
	# model = BaggingClassifier(base_estimator = [RandomForestClassifier(), linear_model.LogisticRegression()])
	# model = EnsembleClassifier([BaggingClassifier(base_estimator = RandomForestClassifier()),
	#                             GradientBoostingClassifier])
	# model = GradientBoostingClassifier()

	feature_columns = df_train.iloc[:, 4:]

	# Train the classifier
	model.fit(feature_columns, df_train.Driver)
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

def calc_prob_iforest(df_features_driver):
	df_train = df_features_driver
	feature_columns = df_train.iloc[:, 3:]
	feature_data = feature_columns.as_matrix()

	# print(df_features_driver.shape[0])
	for i in range(df_features_driver.shape[0]):
		# print(i)
		left_part = feature_data[:i]
		right_part = feature_data[i+1:]

		# print(left_part)
		# print(right_part)

		if i == 0:
			tmp_data = right_part
		else:
			if i + 1 < df_features_driver.shape[0]:
				tmp_data = np.concatenate([left_part, right_part])
			else:
				tmp_data = left_part

		# print(tmp_data)
		# print(i)
		model = iForest.iForest(tmp_data, 100, 20)
	# print(df_features_driver)
		print(model.anomalyScore(feature_data[i]))

def main():
	process_n_drivers = 2736 # 2736 is all drivers, or use -1
	if len(sys.argv) == 2:
		process_n_drivers = int(sys.argv[1])

	features_path = path.join('features')
	features_files = listdir(features_path)

	# Get data frame that contains each trip with its features
	# features_df_list = [pd.read_hdf(path.join(features_path, f), key = 'table') for f in features_files]
	# feature_df = pd.concat(features_df_list)

	#features_df_list_2 = [pd.read_hdf(path.join(features_path_2, f), key = 'table') for f in features_files_2]
	#feature_df_2 = pd.concat(features_df_list_2)
	#feature_df_2x = feature_df_2[['Driver', 'Trip', 'mean_speed_times_acceleration', 'pauses_length_mean']]

	# feature_df = pd.merge(feature_df_1, feature_df_2x, on=['Driver', 'Trip'], sort = False)

	# feature_df.reset_index(inplace = True)
	df_list = []

	for f in features_files:
		f = "features_driver_1352.h5"
		driver_features = pd.read_hdf(path.join(features_path, f), key = 'table')
		# print(driver_features)

		calc_prob_iforest(driver_features)

		process_n_drivers -= 1
		if process_n_drivers == 0:
			break

	# for i, (driver, driver_df) in enumerate(feature_df.groupby('Driver')):
	# 	# indeces = np.append(np.arange(i * 200), np.arange((i+1) * 200, len(feature_df))) # dafuq is this?

	# #     # other_trips = indeces[np.random.randint(0, len(indeces) - 1, 200)]
	# #     # others = feature_df.iloc[other_trips]
	# #     # others.Driver = int(0)
	# 	print("Driver", driver)
	# 	print(driver_df)
	# 	# driver_df = feature_df.loc(1352)
	# 	# submission_df = calc_prob_iforest(driver_df)

	# 	process_n_drivers -= 1
	# 	if process_n_drivers == 0:
	# 		break
	# #     submission_df = calc_prob(driver_df, others)
	# #     df_list.append(submission_df)

	# # submission_df = pd.concat(df_list)
	# # create_submission_file(submission_df)


if __name__ == "__main__":
	main()
