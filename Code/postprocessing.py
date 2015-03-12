import pandas as pd
import numpy as np
from os import listdir

df_submission = pd.read_csv('../calibration/submission-1.csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(suppress=True)
step = 1 / 200

drivers = listdir("../drivers")

new_probs = np.tile(np.arange(0, 1, step), len(drivers))
# new_probs = np.ones(len(df_submission))

print(len(new_probs))
print(len(df_submission))

sorted_df = df_submission.sort(['driver_trip', 'prob'])
# sorted_df['prob'] = new_probs

# df_submission.prob = df_submission.prob.apply(change_prob)

sorted_df['prob'] = sorted_df['prob'].map(lambda x: '%f' % x)
sorted_df.to_csv('submission_processed_AUC.csv', index = False, float_format='%f'.format)
