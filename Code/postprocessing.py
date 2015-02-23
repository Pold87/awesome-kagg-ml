import pandas as pd


df_submission = pd.read_csv('submission-7.csv')

def change_prob(x):
    if (x > 0.4):
        return x
    else:
        return x * 0.2

df_submission.prob = df_submission.prob.apply(change_prob)


df_submission.to_csv('submission_processed.csv', index = False)


