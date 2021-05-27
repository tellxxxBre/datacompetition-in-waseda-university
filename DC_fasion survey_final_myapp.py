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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.cluster import KMeans
from sklearn import tree
from bottle import get, post, request, run

df1 = pd.read_csv("parts-file01.csv", usecols=[13])
df1 = df1[df1['AAAA'] == 11]

df2_1 = pd.read_csv("parts-file02.csv", usecols=[16,17,20,23])

df2_2 = pd.read_csv("parts-file03.csv", usecols=[12,17,20,22,38,41,42,43,48,50,134])

df2_3 = pd.read_csv("parts-file05.csv", usecols=[15,38,41])

df3 = pd.concat([df1, df2_1, df2_2, df2_3], axis=1)
df3 = df3.dropna() 

df_new = df3.drop('AAAA', axis=1)
df_new = df_new.reset_index(drop=True)

ss = StandardScaler()
df_new_s = pd.DataFrame(ss.fit_transform(df_new))

n_components=4
fa = FA(n_components, max_iter=5000)
df_new_pro = fa.fit_transform(df_new_s)
df_new_pro = pd.DataFrame(df_new_pro)
df_new_pro_s = pd.DataFrame(ss.fit_transform(df_new_pro), 
               columns=["第1因子", "第2因子", "第3因子", "第4因子"],
                )
df_new_pro_s_ar = df_new_pro_s.values
km = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
km.fit(df_new_pro_s_ar)
df_new_pro_s_arhat = km.predict(df_new_pro_s_ar)
df_new_pro_clust = df_new_pro[:]

df_new_pro_clust
df_new_pro_clust["ID"] = df_new_pro_s_arhat
train_y = df_new_pro_clust["ID"].values
X = df_new_pro_clust.drop("ID", axis=1).values

@get('/')
def circle_area():
    return '''
        <form action="/calc" method="post">
            質問a~rまでの回答を番号で入力してください。あなたのクラスが表示されます。<br>
            １；あてはまる、２；ややあてはまる、３；あまりあてはまらない、４；あてはまらない<br><br>
            なお、質問oについては、「ファッション関連の新しいモノやコト、サービスなど」は<br>
            １；とにかくすぐに買ったり利用してみる<br>
            ２；流行りそうだと感じたら買ったり利用してみる<br>
            ３；流行り始めたら買ったり利用してみる<br>
            ４；周りのみんなが使うようになってから買ったり利用してみる<br>
            ５；関心がない<br>
            の番号で回答してください。<br><br><br>
            
            a:気に入ったものならば、中古商品やリサイクル商品でも構わない<br>
            <input name="a" type="text" /><br>   
            b:品質・機能を、しっかり比較検討して購入する<br>
            <input name="b" type="text" /><br>
            c:SNSやブログ、サイトなどで推奨されているものを購入することが多い<br>
            <input name="c" type="text" /><br>
            d:多くの人が使用している商品であるかどうかが大切だと思う<br>
            <input name="d" type="text" /><br>
            e:ファッション雑誌をよく読む<br>
            <input name="e" type="text" /><br>
            f:有名人などのファッションや髪型、メイクを参考にする<br>
            <input name="f" type="text" /><br>
            g:常に最新のファッションを身につけたい<br> 
            <input name="g" type="text" /><br>
            h:購入している服や靴のブランドにはこだわりがある<br> 
            <input name="h" type="text" /><br>
            i:ファッションで自分らしさを表現したい<br> 
            <input name="i" type="text" /><br>
            j:ブランドは金額が高いほど品質が良いと思う<br>
            <input name="j" type="text" /><br>
            k:高級ブランドはセールや海外通販などでできるだけ安く手に入れたい<br>
            <input name="k" type="text" /><br>
            l:ベーシックで、無難なファッションが好きな方だ<br> 
            <input name="l" type="text" /><br>
            m:ゴージャスで主張のあるブランド・アイテムを持っておきたい<br> 
            <input name="m" type="text" /><br>
            n:ファストファッションもよく取り入れている<br>
            <input name="n" type="text" /><br>
            o:ファッションイノベーター度<br> 
            <input name="o" type="text" /><br>
            p:異性から魅力的だと思われたい<br> 
            <input name="p" type="text" /><br>
            q:ファッションやおしゃれに気を遣っている<br> 
            <input name="q" type="text" /><br>
            r:ファッションにはこだわりがある<br> 
            <input name="r" type="text" /><br><br>
            <input value="calculate" type="submit" />
        </form>
    '''

@post('/calc')
def circle_area():
    a = request.forms.get('a')
    b = request.forms.get('b')
    c = request.forms.get('c')
    d = request.forms.get('d')
    e = request.forms.get('e')
    f = request.forms.get('f')
    g = request.forms.get('g')
    h = request.forms.get('h')
    i = request.forms.get('i')
    j = request.forms.get('j')
    k = request.forms.get('k')
    l = request.forms.get('l')
    m = request.forms.get('m')
    n = request.forms.get('n')
    o = request.forms.get('o')
    p = request.forms.get('p')
    q = request.forms.get('q')
    r = request.forms.get('r')
    new = [[int(a), int(b), int(c), int(d), int(e), int(f), int(g), int(h), int(i), 
            int(j), int(k), int(l), int(m), int(n), int(o), int(p), int(q), int(r)]]
    newdata = np.array(new)
    newdata = pd.DataFrame(newdata)

    ss.fit(df_new)

    newdata_s = pd.DataFrame(ss.transform(newdata))

    n_components=4
    fa = FA(n_components, max_iter=5000) 
    fa.fit(df_new_s)
    newdata_pro = fa.transform(newdata_s)

    dtc = tree.DecisionTreeClassifier(max_depth=5)
    dtc.fit(X, train_y)

    Z = newdata_pro
    train_yhat_new = dtc.predict(Z)
    
    if train_yhat_new == [0]:
        return "あなたのクラスIDは、" + str(train_yhat_new) + "です。このクラスは無難なファッションやファストファッションを好むグループです。"
    elif train_yhat_new == [1]:
        return "あなたのクラスIDは、" + str(train_yhat_new) + "です。このクラスはファッションに関して低関心なグループです。"
    elif train_yhat_new == [2]:
        return "あなたのクラスIDは、" + str(train_yhat_new) + "です。このクラスはファッションに関して非常に関心があるグループです。"
    elif train_yhat_new == [3]:
        return "あなたのクラスIDは、" + str(train_yhat_new) + "です。このクラスは商品の機能や品質を重視するグループです。"
    else:
        return "あなたのクラスIDは、" + str(train_yhat_new) + "です。このクラスは平均的な大学生のグループです。古着ファッションや特定のブランドへのこだわりが強い傾向があります。"
    
    
run(host='localhost', port=8080, debug=True)