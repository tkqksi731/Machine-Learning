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
# 1.데이터가져오기
# X, y = mglearn.datasets.make_wave(n_samples=60)
X, y = mglearn.datasets.load_extended_boston()
# 2.데이터 조개기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# 단순선형회귀모델 분석
linear_model = LinearRegression()
linear_model.fit( X_train, y_train)

print("\nLinearRegression model -------")
print("train score {:0.2f}".format(
    linear_model.score(X_train, y_train)))
print("test score {:0.2f}".format(
    linear_model.score(X_test, y_test)))


# 리지선형회귀모델 분석
ridge_model = Ridge()
ridge_model.fit( X_train, y_train)

print("\nridge model : alpha -------")
print("train score {:0.2f}".format(
    ridge_model.score(X_train, y_train)))
print("test score {:0.2f}".format(
    ridge_model.score(X_test, y_test)))



ridge_model10 = Ridge(alpha=10)
ridge_model10.fit( X_train, y_train)

print("\nridge model : alpha 10 -------")
print("train score {:0.2f}".format(
    ridge_model10.score(X_train, y_train)))
print("test score {:0.2f}".format(
    ridge_model10.score(X_test, y_test)))



ridge_model01 = Ridge(alpha=0.1)
ridge_model01.fit( X_train, y_train)

print("\nridge model : alpha 0.1 -------")
print("train score {:0.2f}".format(
    ridge_model01.score(X_train, y_train)))
print("test score {:0.2f}".format(
    ridge_model01.score(X_test, y_test)))


#그림그리기 - 특성마다 하나의 coef_값
print( ridge_model.coef_ )
#plt.plot 산포도 plot(x, y)
#plt.plot(y) y축
plt.plot( ridge_model.coef_, '^',label="alpha=1")
plt.plot( ridge_model10.coef_, 's',label="alpha=10")
plt.plot( ridge_model01.coef_, 'v',label="alpha=0.1")

plt.plot( linear_model.coef_, "o", label="linear")

# 차트 이쁘게 가고작업
# x축 제목 붙이기
plt.xlabel("계수목록 (특성) - feature")

# y축 제목 붙이기
plt.ylabel("계수 크기 : 기울기")

# 각 축의 크기 조절하기
xlims = plt.xlim() #현재 x축의 기본값들을 가져온다
#수평라이을 그린다
plt.hlines( 0 , xlims[0], xlims[1])
# y축 간격 조절 - 값의 차이가 너무 커서 특정 값들이 잘 안보임
plt.ylim(-25, 25)
plt.xlim( xlims )
plt.legend() # 범주 출력하기 - 범주
plt.show()