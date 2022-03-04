import pandas as pd

# データセットのロード
AndroidDataset = pd.read_csv('train.csv', sep=';')
print(AndroidDataset.type.value_counts())
print(AndroidDataset.columns)
print(pd.Series.sort_values(
    AndroidDataset[AndroidDataset.type==1].sum(axis=0),
    ascending=False
    )[1:11])

top10 = ['android.permission.INTERNET',
         'android.permission.READ_PHONE_STATE',
         'android.permission.ACCESS_NETWORK_STATE',
         'android.permission.WRITE_EXTERNAL_STORAGE',
         'android.permission.ACCESS_WIFI_STATE',
         'android.permission.READ_SMS',
         'android.permission.WRITE_SMS',
         'android.permission.RECEIVE_BOOT_COMPLETED',
         'android.permission.ACCESS_COARSE_LOCATION',
         'android.permission.CHANGE_WIFI_STATE']

AndroidDataset.loc[AndroidDataset.type==1, top10].sum()
AndroidDataset.loc[AndroidDataset.type==0, top10].sum()
import matplotlib.pyplot as plt
fig, axs =  plt.subplots(nrows=2, sharex=True)

AndroidDataset.loc[AndroidDataset.type==0, top10].sum().plot.bar(ax=axs[0])
AndroidDataset.loc[AndroidDataset.type==1, top10].sum().plot.bar(ax=axs[1], color="red")

from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
import csv
import random

# X,yに特徴量とラベルをそれぞれ代入
X = AndroidDataset.iloc[:,:-1]
y = AndroidDataset.iloc[:, -1]
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate

# データセットをテスト用のデータに20%を割り当てて分割
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, shuffle=True, random_state=101)

# SVMのハイパーパラメータチューニング用のクラスを設定
class Objective_SVM:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # チューニング対象のパラメーターを指定
        params = {
            'kernel': trial.suggest_categorical(
                'kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': trial.suggest_loguniform('C', 1e-5, 1e2),
            'gamma': trial.suggest_categorical(
                'gamma', ['scale','auto']),
        }
        # モデルの初期化
        model = SVC(**params)

        scores = cross_validate(model,
                                X=self.X, y=self.y,
                                n_jobs=-1)
        return scores['test_score'].mean()

# チューニングの対象クラスを設定
objective = Objective_SVM(X_train, y_train)
study = optuna.create_study(direction='maximize')
# 最大で1分間チューニングを実行
study.optimize(objective, timeout=60)
# ベストのパラメーターの出力
print('params:', study.best_params)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 探索結果として得られたベストのパラメーターを設定

model = SVC(
    kernel = study.best_params['kernel'],
    C = study.best_params['C'],
    gamma = study.best_params['gamma']
)
# モデルの訓練
model.fit(X_train, y_train)
# テスト用のデータを使用して予測
pred = model.predict(X_test)
# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))