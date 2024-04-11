import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm

df=pd.read_csv("seeds_dataset.csv",header=None)

df.drop(df.columns[[7,8,9]],axis=1,inplace=True) #7,8,9列目の削除
df.dropna(how='any',inplace=True) #Nanが含まれる行の削除

scaler=StandardScaler() #標準化を行う関数StandardScalerをscalerという名前に定義
dfs=scaler.fit_transform(df) #データであるdfを標準化

corrs = np.corrcoef(dfs,rowvar=False) #相関係数の計算
sns.heatmap(corrs, cmap=sns.color_palette('Reds', 10)) #ヒートマップで可視化

pca=PCA()
pca.fit(dfs) #主成分分析
ev = pd.DataFrame(pca.explained_variance_ratio_) #寄与率
t_ev = pd.DataFrame(pca.explained_variance_ratio_.cumsum()) #累積寄与率

#可視化に用いる変数定義
length=7
kiyoritu=ev*100
ruiseki=t_ev*100
xlab=np.array([1,2,3,4,5,6,7])

# サイズ指定
fig = plt.figure()
# 軸関係の表示
ax = fig.add_subplot(111)

# データ数のカウント
data_num=length

# 棒グラフの描画
ax.bar(range(data_num), kiyoritu[0])
ax.set_xticks(range(data_num))
ax.set_xticklabels(xlab)
ax.set_ylim([0,100])

# 折れ線グラフの描画
ax_add = ax.twinx()
ax_add.plot(range(data_num), ruiseki[0])
ax_add.set_ylim([0, 100])
plt.show()

pcs=pca.transform(dfs) #転置
pcs2=pcs[:,:2] #2列目までを抽出（第二主成分までを抽出）

km = KMeans(n_clusters=3,            # クラスターの個数         # セントロイドの初期値をランダムに設定  default: 'k-means++'
            n_init=10,               # 異なるセントロイドの初期値を用いたk-meansの実行回数 default: '10' 実行したうちもっとSSE値が小さいモデルを最終モデルとして選択
            max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数  default: '300'
            tol=1e-04,               # 収束と判定するための相対的な許容誤差 default: '1e-04'
            random_state=0)          # セントロイドの初期化に用いる乱数発生器の状態

cluster = km.fit_predict(pcs2) #クラスタリング
print(cluster)

plt.scatter(pcs2[cluster==0,0],         # cluster（クラスター番号）が0の時にpcs2の0列目を抽出
                    pcs2[cluster==0,1], # cluster（クラスター番号）が0の時にpcs2の1列目を抽出
                    s=50,
                    label='cluster 1')
plt.scatter(pcs2[cluster==1,0],
                    pcs2[cluster==1,1],
                    s=50,
                    label='cluster 2')
plt.scatter(pcs2[cluster==2,0],
                   pcs2[cluster==2,1],
                    s=50,
                    label='cluster 3')
plt.legend() #凡例
plt.grid() #グリッド線
plt.show()

cluster_labels = np.unique(cluster)       # clusterの要素の中で重複を無くす
n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定した3となる

# シルエット係数を計算
silhouette_vals = silhouette_samples(pcs2,cluster,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算
y_ax_lower, y_ax_upper= 0,0
yticks = []

for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[cluster==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
        color = cm.jet(float(i)/n_clusters)               # 色の値を作る
        plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
                         c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                         height=1.0,                      # 棒の高さ
                         edgecolor='none',                # 棒の端の色
                         color=color)                     # 棒の色
        yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
        y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く
plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
plt.ylabel('Cluster')
plt.xlabel('silhouette coefficient')
plt.show()

PC1 = pca.components_[0,:] #第一主成分
PC2 = pca.components_[1,:] #第二主成分
ldngs = pca.components_
scalePC1 = 1.0/(PC1.max() - PC1.min())
scalePC2 = 1.0/(PC2.max() - PC2.min())
features = ["Area","Perimeter","Compactness",
    "Length of kernel","Width of kernel","Asymmetry Coeff",
    "Length of kernel groove"] #各変数名

fig, ax = plt.subplots(figsize=(14, 9))

for i, feature in enumerate(features):
    ax.arrow(0, 0, ldngs[0, i],
             ldngs[1, i])
    ax.text(ldngs[0, i] * 1.15,
            ldngs[1, i] * 1.15,
            feature, fontsize=18)

ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
ax.set_title('Figure 1', fontsize=20)
plt.show()