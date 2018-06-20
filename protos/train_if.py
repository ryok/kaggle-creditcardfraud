import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("../input/creditcard.csv")
df_org = df.copy()

del df['Time']
del df['Amount']
del df['Class']

#print(df[:10])

iforest = IsolationForest()
iforest.fit(df)
iforest_result_value = iforest.decision_function(df)

#print(iforest_result[:10])

iforest_result = df_org.copy()
iforest_result['if'] = iforest_result_value

iforest_top1000 = iforest_result.sort_values(by=['if'], ascending=True)[:1000]

print(iforest_top1000)
