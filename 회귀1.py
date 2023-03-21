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

#회귀
reg = KNeighborsRegressor(n_neighbors=1)
reg.fit(X_train, y_train)

# 이웃하고 비교

y_pred = reg.predict( X_test )
print(y_pred)
print(y_test)

print("정확도 : {}".format(reg.score(X_train, y_train)))
print("테스트정확도 : {}".format(reg.score(X_test, y_test)))