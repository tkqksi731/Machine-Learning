import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets 
import mglearn 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
#분류  - k이웃알고리즘 
#차트가 한글지원을 하지 않아서 한글지원하게 
import matplotlib
import matplotlib.font_manager as fm 

font_path = "C:/Windows/Fonts/malgun.TTF"
font_name = fm.FontProperties(fname=font_path).get_name()
#-안나오는 문제 처리 
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rc('font', family=font_name)

#------- 공통사항 ---------------------------
#1.데이터가져오기
#X, y = mglearn.datasets.make_wave(n_samples=60)
X, y = mglearn.datasets.load_extended_boston()

#2.데이터 쪼개기 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, random_state=0
)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso 

#3.선형회귀모델 분석
linear_model = LinearRegression()
linear_model.fit( X_train, y_train)

print("\nLinearRegression model ------------")
print("train score {:.2f}".format(
         linear_model.score(X_train, y_train)))
print("test  score {:.2f}".format(
         linear_model.score(X_test, y_test)))
#라쏘
lasso_model = Lasso()
lasso_model.fit( X_train, y_train)

print("\nRidge model ------------")
print("train score {:.2f}".format(
         lasso_model.score(X_train, y_train)))
print("test  score {:.2f}".format(
         lasso_model.score(X_test, y_test)))
count = np.sum( lasso_model.coef_!=0 )
print("사용하는 특성의 개수 : ", count)
print( lasso_model.coef_ )

#알파값을 바꿀때 라소는 max_iter 값을 줘야 한다 
lasso_model01 = Lasso(alpha=0.01, max_iter=10000)
lasso_model01.fit( X_train, y_train)

print("\nRidge model : alpha 0.01 ------------")
print("train score {:.2f}".format(
         lasso_model01.score(X_train, y_train)))
print("test  score {:.2f}".format(
         lasso_model01.score(X_test, y_test)))
count = np.sum( lasso_model01.coef_!=0 )
print("사용하는 특성의 개수 : ", count)
print( lasso_model01.coef_ )


lasso_model0001 = Lasso(alpha=0.0001, max_iter=10000)
lasso_model0001.fit( X_train, y_train)

print("\nRidge model : alpha 0.0001 ------------")
print("train score {:.2f}".format(
         lasso_model0001.score(X_train, y_train)))
print("test  score {:.2f}".format(
         lasso_model0001.score(X_test, y_test)))
count = np.sum( lasso_model0001.coef_!=0 )
print("사용하는 특성의 개수 : ", count)
print( lasso_model0001.coef_ )

#그림그리기 - 특성마다 하나의 coef_값 
print( lasso_model.coef_  )
#plt.plot 산포도  plot(x, y)
#plt.plot(y) y축 
plt.plot( lasso_model.coef_,   '^', label="alpha=1")
plt.plot( lasso_model01.coef_, 's', label="alpha=10")
plt.plot( lasso_model0001.coef_, 'v', label="alpha=0.1")

plt.plot( linear_model.coef_ , 'o', label='linear')

#차트 이쁘게 가공작업 
#x축 제목 붙이기 
plt.xlabel("계수목록(특성)-feature")

#y축 제목 붙이기 
plt.ylabel("계수 크기 : 기울기 ")

#각 축의 크기 조절하기 
xlims = plt.xlim() #현재 x축의 기본값들을 가져온다 
#수평라인을 그린다 
plt.hlines( 0, xlims[0], xlims[1])
#y축 간격 조절 - 값의 차이가 너무 커서 특정 값들이 잘 안보임
plt.ylim(-25, 25)
plt.xlim( xlims )
plt.legend() #범주 출력하기 - 범주 
plt.show() 
