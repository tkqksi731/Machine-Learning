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

# y = wx + b : w와 b를 찾아내는 과정이 회귀
# 특성이 여러개일때
#y = w1x1 + w2xs2 + w3x3 + w4x4
data, target = mglearn.datasets.make_wave(n_samples=40)
print(data.shape)
print(target.shape)

print(data[:10])
print(target[:10])

# 분류는 KNeighborsClassifier 사용
# 회귀는 KNeighborsRegressor를 사용한다
from sklearn.neighbors import KNeighborsRegressor

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=0)

from sklearn.linear_model import LinearRegression

# 선형회귀모델 만들기
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
for i in range(0, len(y_pred)):
    print(y_pred[i], y_test[i], y_pred[i]==y_test[i])

print("기울기 : ", model.coef_)
print("절편 : ", model.intercept_)

#학급성과평가
print("훈련셋 {}".format(model.score(X_train, y_train)))
print("테스트 {}".format(model.score(X_test, y_test)))
