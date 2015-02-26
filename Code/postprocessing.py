import pandas as pd

df_submission = pd.read_csv('submission-1.csv')

def change_prob(x):
    if (x > 0.59):
        return 1
    else:
        return 0

df_submission.prob = df_submission.prob.apply(change_prob)


df_submission.to_csv('submission_processed_2.csv', index = False)
