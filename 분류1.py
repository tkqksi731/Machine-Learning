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

#책 저자가 만든 데이터 생성 함수
X, y = mglearn.datasets.make_forge()
print( X.shape, type(X) ) # 26 by 2, ndarray
print( y.shape, type(y) ) # 26 by 1, ndarray

print(X[:5])
print(y[:5]) # 1 0 1 0 0

# 데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 학습하기
clf = KNeighborsClassifier(n_neighbors=1)
#끼리끼지 모인다. 학습을 하고 나면 입력데이터들이 묶인 공간이
#있으면 새로 입력된 값이 자기 가장 가까운 이웃을 찾아서
#내가 그 이웃과 같다고 생각한다
#이웃이 하나일 경우에는 훈련셋에 맞춰서 과대적합 되는경우가
#많다 보통은 이웃을 숫자를 3~5개로 쓰는 경우가 많다.

#훈련셋트 입력과 출력을 넣고 학습을 한다
clf.fit(X_train, y_train)
# 성능평가 - {:.2} 소수점 이하 2자리까지만 표출
print("훈련세트 {:.2f}".format(clf.score(X_train, y_train)))
print("테스트세트 {:.2f}".format(clf.score(X_test, y_test)))

#실제 예측 - 벡터를 입력받아서 벡터를 출력하한다
y_pred = clf.predict( X_test )
print("입력값 : ", X_test )
print("예측값 : ", y_pred )
print("실제값 : ", y_test )


fig, axes = plt.subplots(1,3,figsize=(10,3))
print(fig, axes)

# 이웃숫자 리스트 만들기
k_neighors = [1,2,3] # 이웃숫자가 1일대 , 2일떄, 3일때
for n_neighbors, ax in zip( k_neighors, axes):
    #이웃이 하나일때 1번 차트
    #이웃이 둘일때 2번차트
    #이웃이 셋일때 3번차트에 그리겠다는 의미
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X,y) #학습

    #각각 차트 그리기
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    #학습된 내용으로 특성 분포도 그리기
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)

    #표의 제목줄
    ax.set_title("{} 이웃".format(n_neighbors))
    ax.set_xlabel("특성 0") # x축 제목
    ax.set_ylabel("특성 1") # y축 제목
axes[0].legend(loc=3)
plt.show() # vs코드에서는 써야함 쥬피터는 사용 x