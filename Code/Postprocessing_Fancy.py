import pandas as pd
import numpy as np




df_normal = pd.read_csv('submission_processed_AUC.csv')
df_matched = pd.read_csv('submission-matched-0.csv')


s1 = df_normal.sort(['driver_trip'])
s2 = df_matched.sort(['driver_trip'])


submission = pd.DataFrame(columns=['driver_trip', 'prob'])
submission['driver_trip'] = s1.driver_trip
submission['prob'] = s1.prob * s2.prob


submission.to_csv("submission-super-fancy.csv", index=False)