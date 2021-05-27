"""
今回の調査では大学生を対象とすることにした
"""
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

#対象は職種で大学生・大学院生と選んだ人
#大学生・大学院生を抽出（１０５８人）
df1 = pd.read_csv("parts-file01.csv", usecols=[13])
df1 = df1[df1['AAAA'] == 11]

#関心がある話題の質問項目を抽出
df2 = pd.read_csv("parts-file05.csv", usecols=[*range(320, 347)])

df3 = pd.concat([df1, df2], axis=1)
df3 = df3.dropna() #欠損値の削除

df_new = df3.drop('AAAA', axis=1)
df_new = df_new.reset_index(drop=True)

df_new = np.sum(df_new, axis=0)
left = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
plt.bar(left, df_new)

"""
上の棒グラフを見ると5,6,7,8,13,17に対して関心が高いことがわかる
今回のADKアンケート調査によると、ファッション(13)に関するアンケート質問項目が非常に多かったので
今回の分析の題目は大学生のファッション価値観を分類するという目的とする
"""