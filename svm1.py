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
# 1.데이터 가져오기 - 분류데이터 가져오기
cancer = datasets.load_breast_cancer()

#데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    cancer["data"], cancer["target"], random_state=0
)

# 회귀 분류 둘다 가능
#서포트 벡터 머신 : 기준이 선형 회귀 모델들이
# 2차원에 선을 그어서 분류를 했다면, 서포트벡터머신은
# 3차원에 선을 그어서 공간에 선을 그어서 분류를 한다
# 분류를 잘하는데 특성중에 단위가 다를 경우
# 분류가 잘 안된다. 스케일링(단위 맞추기)
# 특히나 서포트벡터머신은 반드시 데이터전처리(4장) 작업을 해야한다
from sklearn.svm import SVC
forest = SVC()
forest.fit(X_train, y_train)

print("train {}".format(forest.score(X_train, y_train)))
print("test {}".format(forest.score(X_test, y_test)))
# 훈련셋과 데이터셋이 차이가 너무 크다 - 과대적합상태

#데이터 - 데이터최소값/범위로 나눈다
def scaling(value, min_value, range_value):
    value = (value-min_value)/range_value
    return value

# axis=0 행, axis=1 열
min_value = X_train.min(axis=0)
range_value = (X_train - min_value ).max(axis=0)
print( min_value )
print( range_value )

X_train_scaled = scaling(X_train, min_value, range_value )
print( X_train_scaled )

print("특성별 최소값 : ", X_train_scaled.min(axis=0))
print("특성별 최대값 : ", X_train_scaled.max(axis=0))

X_test_scaled = scaling(X_test, min_value, range_value)

svc = SVC()
svc.fit( X_train_scaled, y_train )

print("train {}".format(svc.score(X_train_scaled, y_train)))
print("test {}".format(svc.score(X_test_scaled, y_test)))