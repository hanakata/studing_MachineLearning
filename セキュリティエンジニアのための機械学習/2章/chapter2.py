# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:23:12 2022

@author: katayama
"""

#インポート
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

#データロード
training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

#変数Xに最終列以外すべての列を、yには最終列を代入
X = training_data[:,:-1]
y = training_data[:, -1]

#scikit-learnのtrain_test_splitを利用してデータセットを訓練用とテスト用に分割する
#テスト用にはデータセットの20%を割り当てる。
#これをホールドアウト検証と呼ぶ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=101)

classifier = LogisticRegression(solver='lbfgs')

# 訓練用データを使って検出器を訓練する。
classifier.fit(X_train, y_train)
# 予測させる。
predictions = classifier.predict(X_test)
# このフィッシング検出器の正解率を出力させる。
accuracy = 100.0 * accuracy_score(y_test, predictions)
print("The accuracy of your Logistic Regression on testing data is: {}".format(accuracy))

# 交差検証(5分割)による汎化性能の評価
scores = cross_val_score(classifier, X_train, y_train, cv=5)
# 評価結果の出力
print("Evaluated score by cross-validation(k=5): {}".format(100 * scores.mean()))