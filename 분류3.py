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

# 데이터 만들기
#1. 입력데이터 : 1...(2000)   [ [1], [2]....]
#2. 출력데이터 : 0 0 0 1 0 0 0 1 ...... 2000

limit = 20001
#1. 입력데이터
data = list()
for year in range(1, limit):
    data.append( [year])
print( data[ :10]) #10개만 확인해보자

#2. 출력데이터: 윤년구하는 함수 윤년이면 1, 아니면 0을 반환하는 함수
def isLeap(year):
    if year%4==0 and year%100!=0 or year%400==0:
        return 1
    else:
        return 0

#파이썬은 벡터가 입력되면 벡터를 내보낸다.
target = list()
for i in range(1, limit):
    target.append(isLeap(i))

#머신러닝에서는 ndarray타입으로
#nd.array타입으로 전환
data = np.array(data)
target = np.array(target)
print(data.shape, target.shape)

#데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=0)
clf = KNeighborsClassifier(n_neighbors=1)


print(X_train.shape, X_test.shape)
print(X_train[:10], X_test[:10])

print(y_train.shape, y_test.shape)
print(y_train[:10], y_test[:10])

for i in range(1,11):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    """
    y_pred = clf.predict(X_test)
    for i in range(0, 50):
        print("실제값:", y_train[i], 
            "예측값:", y_pred[i],
            y_train[i]==y_pred[i])
    """
    print("훈련데이터셋 : {:.2f}".format(
        clf.score(X_train, y_train)))
    print("테스트데이터셋 : {:.2f}".format(
        clf.score(X_test, y_test)))