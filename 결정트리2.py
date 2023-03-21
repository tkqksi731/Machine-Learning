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

font_path = "C:/Windows/Fonts/malgun.TTF"
font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc("font", family=font_name)
# - 안나오는 문제 처리
matplotlib.rcParams['axes.unicode_minus'] = False  
# ---------------------공통 사항---------------------------
#1.데이터 가져오기-분류데이터
import os #윈도우 os제어 라이브러리
#파일을 읽어올 때 절대경로나 상대경로로 가능하다
#절대경로 mglearn.datasets.DATA_PATH - 상수
#본인이 데이터 파일이 있는 경로가 

path = "c:/딥러닝/머신러닝/data/ram_price.csv"
#상대경로는 프로그램이 가동하는 폴더를 .로 표시
ram_price = pd.read_csv("./data/ram_price.csv", encoding="cp949", header=3, index_col=False)
        #index_col = False 자동 인덱스 막기
        #header = 정수 헤더가 있는 라인수
        #encoding = 엑셀은 cp949, 메모장류는 utf8
print( ram_price)
#컬럼명 확인
print( ram_price.columns)
#컬럼명 변경
ram_price.columns=["id", "date", "price"]
print( ram_price.columns)
#열을 삭제한다, drop 함수는 열이나 행을 삭제할 수 있다
#열을 삭제할 경우에 drop("삭제할 컬럼명", axis=1)
#axis =1 -> 열을 삭제하라는 의미다. 행일 경우에는 axis=0이다
#특정 칼럼을 자체저그로 삭제하는 것이 아니라 삭제한 나머지를 반환하는 특이한 구조이다.
#만일 자신의 칼럼을 보관하지 않고 삭제하려면 반환값을 다시 받아야 한다.
data = ram_price.drop("id", axis=1)
print(data.head()) #맨앞에서 5개 데이터만 확인하는 함수
print(ram_price.head())

plt.semilogy( ram_price.date, ram_price.price)
plt.show()

#결정트리 회귀 - 문제점이 있음 없는 기간은 예측 불가능
from sklearn.tree import DecisionTreeRegressor
#2000 년 이하 자름 : 문제점을 보여주려고
data_train = ram_price[ram_price.date<2000]
data_test = ram_price[ram_price.date>=2000]
print( data_train.head())
print( data_test.head())

#1차원 -> 2차원으로 바꾸기
X_train = data_train.date[:, np.newaxis]
print( type(X_train)) 
#간단하게 하기 위해 np.log를 써서 스케일을 줄임
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor()
tree.fit( X_train, y_train)

from sklearn.linear_model import LinearRegression
lrg = LinearRegression()
lrg.fit(X_train, y_train)

#그리고 예측은 전체 데이터에 한다.
X_all = ram_price.date[:, np.newaxis]

#디시전트리를 이용한 예측
y_tree = tree.predict( X_all)
#선형회귀를 이용한 예측
y_lrg = lrg.predict(X_all)

#단위 바꾼거 원상 복구
y_tree = np.exp( y_tree )
y_lrg = np.exp(y_lrg)

plt.semilogy(data_train.date, data_train.price, label='훈련데이터')
plt.semilogy(data_test.date, data_test.price, label='테스트 데이터')
plt.semilogy(ram_price.date, y_tree, label='결정트리')
plt.semilogy(ram_price.date, y_lrg, label='선형회귀')
plt.legend() #범주 그리라고 하자
plt.show()

# 차트그리기 중요!
# def treeChart(feature_names, feature_importances):
#     #특성의 개수
#     n_feature = len(feature_importances)
#     #수평 막대그래프
#     plt.barh(np.arange(n_feature), #특성개수만큼 정수배열
#         feature_importances,
#         align="center" )
#     #y축 눈금 제목
#     plt.yticks(np.arange(n_feature), feature_names)
#     plt.ylim(-1, n_feature)
#     plt.show()

# treeChart(np.array(cancer["feature_names"]),
#             tree.feature_importances_)
