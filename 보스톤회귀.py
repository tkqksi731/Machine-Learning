import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#분류 - k이웃알고리즘

#차트가 한지원을 하지 않아서 한글지원하게
import matplotlib
import matplotlib.font_manager as fm

font_path = "C:/Windows/Fonts/H2GTRM.TTF"
font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc("font", family=font_name)

# ---------------------공통 사항---------------------------

# 문제 : 보스톤 집값 예측 데이터
from sklearn.datasets import load_boston

boston = datasets.load_boston()

X, y = mglearn.datasets.load_extended_boston_boston()

# print("보스턴 키값 : ", boston.keys())

data, target = mglearn.datasets.make_wave(n_samples=40)
print(data.shape)
print(target.shape)

from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(
    boston["data"], boston["target"], random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
for i in range(0, len(y_pred)):
    print(y_pred[i], y_test[i], y_pred[i]==y_test[i])

print("기울기 : ", model.coef_)
print("절편 : ", model.intercept_)

print("훈련셋 {}".format(model.score(X_train, y_train)))
print("테스트 {}".format(model.score(X_test, y_test)))

# y = 기울기 * x + 절편 + 오차
print("오차 :", model._residues)

# 고차항일때 개선책 - Ridge: 선형회귀모델
# 모든 기울기를 0에 가깝게 구하도록 강제 제한을 가한다(L2제약)

from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)

print("훈련셋 {}".format(model.score(X_train, y_train)))
print("테스트 {}".format(model.score(X_test, y_test)))
