"""
今回の分析では大学生・大学院生のファッションに対する価値観をいくつかのクラスに分類する
対象者に行った18の質問（消費意識、ファッション意識、生活意識価値観）をもとに、対象者をいくつかのクラスに分ける
詳しくは、Evernote⑤-0に書いた
18の質問の選び方についてはEvernote⑤-1に書いた
（追記）因子数を5にしたとき、因子の特徴が２、３、５で見分けがつかなくなってしまったので、因子数３に変更する
（追記１）因子数3の場合だと２と3の違いがよくわからん
（追記２）因子数4の場合だとうまくいきそう
（追記３）質問を一つ増やし１８にしてみる（因子数４、クラスター５で分析）
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

#消費意識の質問項目を抽出
df2_1 = pd.read_csv("parts-file02.csv", usecols=[16,17,20,23])
#ファッション意識の質問項目を抽出
df2_2 = pd.read_csv("parts-file03.csv", usecols=[12,17,20,22,38,41,42,43,48,50,134])
#生活意識・価値観の質問項目を抽出
df2_3 = pd.read_csv("parts-file05.csv", usecols=[15,38,41])

#0列目に大学生・大学院生、1列目以降に質問項目
df3 = pd.concat([df1, df2_1, df2_2, df2_3], axis=1)
df3 = df3.dropna() #欠損値の削除
len(df3)

#"AAAA"の列を削除し、Index番号をふる
df_new = df3.drop('AAAA', axis=1)
df_new = df_new.reset_index(drop=True)

#見やすいようにrenameする
df_new = df_new.rename(columns={
    "CBAA_13":"a",
    "CBAA_14":"b",
    "CBAA_17":"c",
    "CBAA_20":"d",
    "CCAB_1":"e",
    "CCAB_6":"f",
    "CCAB_9":"g",
    "CCAB_11":"h",
    "CCAB_27":"i",
    "CCAB_30":"j",
    "CCAB_31":"k",
    "CCAB_32":"l",
    "CCAB_37":"m",
    "CCAB_39":"n",
    "CCAC":"o",
    "CEAA_12":"p",
    "CEAA_35":"q",
    "CEAA_38":"r",
}
)

df_new


"""
設問数が18と多いので設問を縮約する
そのために因子分析を行う
まずは、因子数を決定する
"""
#データの標準化を行い、異なる変数間で大小が評価できるようにする
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
df_new_s = pd.DataFrame(ss.fit_transform(df_new))
df_new_s = df_new_s.rename(columns={
    0:"a",
    1:"b",
    2:"c",
    3:"d",
    4:"e",
    5:"f",
    6:"g",
    7:"h",
    8:"i",
    9:"j",
    10:"k",
    11:"l",
    12:"m",
    13:"n",
    14:"o",
    15:"p",
    16:"q",
    17:"r",
}
)

df_new_s


"""
因子数を決定するために、主成分分析によって求めることができる固有値を計算する
"""
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(df_new_s)

#固有値を計算する
ev = pca.explained_variance_

pd.DataFrame(ev, 
             index=["PC{}".format(x + 1) for x in range(len(df_new.columns))], 
             columns=["固有値"])


"""
計算された固有値をもとにスクリープロットを描く
"""
#固有値1の基準線
ev_1 = np.ones(18)

# 変数を指定
plt.plot(ev, 's-')   # 主成分分析による固有値
plt.plot(ev_1, 's-') # ダミーデータ

# 軸名を指定
plt.xlabel("Number of factors")
plt.ylabel("Eigenvalue")

plt.grid()
plt.show()


"""
因子数の決定基準はいくつかあるが、まずガットマン基準（固有値が１以上の因子を採用）を使って推定する
その場合、上のスクリープロットから因子数は3とすることができる
ガットマン基準により因子数を推定できたが、一応累積寄与率が60%以上となる因子数もみてみる
"""
#寄与率の取得
evr = pca.explained_variance_ratio_

#見やすいように、行名･列名を付与してデータフレームに変換
pd.DataFrame(evr, 
             index=["PC{}".format(x + 1) for x in range(len(df_new.columns))], 
             columns=["寄与率"])


#寄与率の累積値をプロット
plt.plot([0] + list(np.cumsum(evr)), "-o")

plt.xlabel("Number of PC") #主成分の数
plt.ylabel("Cumsum")  #累積寄与率

plt.grid()
plt.show()


"""結果；
・ガットマン基準；4
・寄与率60%以上で因子を採用；5
今回は、因子数４で分析してみる
"""


"""
続いて、得られた結果をもとに因子分析を行う
"""
#sklearnのFactorAnalysis(因子分析)クラスをインポート
from sklearn.decomposition import FactorAnalysis as FA

#因子数を指定
n_components=4

#因子分析の実行
fa = FA(n_components, max_iter=5000) #モデルを定義
df_new_pro = fa.fit_transform(df_new_s) #fitとtransformを一括処理

#737人の因子得点の行列
print(df_new_pro)
print(df_new_pro.shape)


#因子負荷量行列の取得（共通因子が各変数に与える影響度）
print(fa.components_.T)


"""
18の設問に対する各因子の解釈を行う
各因子の解釈はEvernote⑤-2に書いた
"""
#変数Factor_loading_matrixに格納
Factor_loading_matrix = fa.components_.T

#因子の解釈をやりやすくするため、データフレームに変換
Factor_loading_matrix = pd.DataFrame(Factor_loading_matrix, 
             columns=["第1因子", "第2因子", "第3因子", "第4因子"], 
             index=[df_new_s.columns])

Factor_loading_matrix


"""
解釈が行いやすいようにグラフでも見てみる
縦が因子負荷量、横が質問番号
"""
import japanize_matplotlib
Factor_loading_matrix.plot(figsize=(9, 9))


"""
続いて、新たに得られた因子得点のデータをもとに大学生をいくつかのクラスターに分類する
クラスター分析を行う
"""
#まずは各個体の因子得点をデータフレームで再度確認する
df_new_pro = pd.DataFrame(df_new_pro, 
             columns=["第1因子", "第2因子", "第3因子", "第4因子"], 
            )
df_new_pro


#他の情報も確認
df_new_pro.describe()


"""
因子得点のまま、クラスター分析を行う方法と、因子得点のデータを標準化してクラスター分析を行う方法があるが、
今回は後者を採用
前者の場合とも比較してみる
"""
#データの標準化を行い、各質問の情報量の統一を行う
from sklearn import preprocessing

ss = preprocessing.StandardScaler()
df_new_pro_s = pd.DataFrame(ss.fit_transform(df_new_pro), 
               columns=["第1因子", "第2因子", "第3因子", "第4因子"],
                )
#標準化した後の因子得点行列
df_new_pro_s


"""
クラスタ分析を行い、大学生・大学院生のファッション価値観の特徴を分類する
階層的クラスター分析でモデルの可視化をすることでクラスター数を決定しようと思う
"""
#まずは適当なクラスター数を探す（ただし、これは恣意的になることに注意）
from scipy.cluster.hierarchy import linkage, dendrogram

#ユークリッド距離、ウォード法で階層的クラスター分析を行う
df_new_pro_s_hclust = linkage(df_new_pro_s, metric="euclidean", method="ward")
plt.figure(figsize=(12, 8))
dendrogram(df_new_pro_s_hclust)
plt.savefig('figure_1.png')
plt.show()


"""
上の階層を見ると5つのクラスだと切れ味よく分類することができると考えた
故に、クラス数を5として非階層クラスター分析を行う（データは標準化されたもの）
その際、kmeans++法を使おうと思う（kmeansではクラスタ中心が初期値に依存するが、kmeans++では初期値に依存しないという点がある）
kmeans法でも試してみる
"""
from sklearn.cluster import KMeans

#pandasからnumpyの行列に変更し、行列を出力
df_new_pro_s_ar = df_new_pro_s.values

#クラスターの数は5とする、またkmeans++を使う
km = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
km.fit(df_new_pro_s_ar)
#結局、予測されたクラスIDは？
df_new_pro_s_arhat = km.predict(df_new_pro_s_ar)
print(df_new_pro_s_arhat)


"""
この結果からどのクラスにも同じくらい人がいることが分かる
"""
#割り当てられた結果をはじめのデータ(df_new_pro)にクラスターidとして一列追加して出力
df_new_pro_clust = df_new_pro[:]
df_new_pro_clust["ID"] = df_new_pro_s_arhat

#各クラスターの数を出力
print(df_new_pro_clust["ID"].value_counts())

#標準化前の因子得点行列にクラスターIDを付け加えた行列
df_new_pro_clust


"""
続いて回答の偏りがどこに出たのか調べる
因子得点の合計値を見る
"""
#クラスターIDでグループ化し数値を書き出す
df_new_pro_clust_gp = df_new_pro_clust.groupby("ID")

#そして、グループ別に各因子得点の合計を出す
df_new_pro_clust_gp_sm =df_new_pro_clust_gp.sum().T

#各グループの最大値を黄色でマーキング
df_new_pro_clust_gp_sm.style
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
#各グループの最小値は緑色でマーキング
def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: green' if v else '' for v in is_min]

df_new_pro_clust_gp_sm.style.apply(highlight_max)

#ちなみに縦が４つの因子、横がクラスID


df_new_pro_clust_gp_sm.style.apply(highlight_min)


"""
このデータの見方；
クラス０は第2因子に（正に）、第1因子に（負に）影響されている
クラス１は主に第1因子に（負に強く）影響されている
クラス２は第1因子に（正に）、第3因子に（負に）影響されている
クラス３は主に第3因子に（正に強く）影響されている
クラス４は第4因子は（正に）、第2因子に（負に）影響されている
"""


#棒グラフでも見てみる
#その際に絶対値に変換する
df_new_pro_clust_gp_sm = df_new_pro_clust_gp_sm.abs()
df_new_pro_clust_gp_sm.style.bar(color="#4285F4")


"""
最後に、決定木を作成することで、新しいデータから各クラスへ分類する過程を見る
"""
##クラスターIDが付与されたデータを訓練用データとする
# train_y = np.array(df_new_pro_clust["ID"].values)
# print(train_y)

train_y = df_new_pro_clust["ID"].values
print(train_y)

#IDを一旦削除したデータ
#これはdf_new_proと同じ
X = df_new_pro_clust.drop("ID", axis=1).values
print(X)


#決定木で統計モデルを作成

from sklearn import tree
#木の深さは５に設定
dtc = tree.DecisionTreeClassifier(max_depth=5)
dtc.fit(X, train_y)

#モデルの正答率は？
train_yhat = dtc.predict(X)

#予測前のクラスID
print(train_y)
#決定木モデルで予測されたクラスID
print(train_yhat)

#精度
print(dtc.score(X, train_y))     


#決定木モデルの可視化
from sklearn.tree import plot_tree

plt.figure(figsize=(30, 30))
plot_tree(dtc, filled=True)


"""
分析からわかること；
・何か物を売る際にどの価値観を持つ消費者（大学生大学院生）をターゲットにするのか絞りやすい
・あなた（対象は大学生大学院生）がどういったファッションへの意識を持ち、どんな消費行動をし、どんな生活価値観があるのかについての質問をすることで７つのクラスのいずれかに分類することができる
・例えば、早稲田付近に洋服店をオープンする予定だとする
大学生のファッションへの価値観が、いくつかのクラスに分類されることが分かれば、どういった層に向けた洋服店を作るのかターゲットが絞りやすくなる
・大学生が多いある地域に、ある層向けの洋服店がなければ、その層向けの店を作れば儲かるかもしれない
"""
