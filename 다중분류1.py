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
# 1.데이터가져오기 - 분류데이터
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
print( X.shape )
print( X[:10])
print( y.shape )
print( y[:10])

# 2.데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

#분류 모듈
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

svc = LinearSVC()
# 모든 데이터에 대해서 처리
svc.fit(X,y)

print( svc.coef_ ) #기울기들 3 by 2 분류기 3개가 만들어짐
print(svc.intercept_ ) # 절편

#그래프 : 책저자 mglearn
#산포도 그려줌
mglearn.discrete_scatter( X[:, 0], X[:, 1], y)
# 공간을 부할한다. -15, 15, 간격개수만큼, 분할, array
#만든다, 기본이 100개 스텝으로 나눔
line = np.linspace(-15, 15, 100)
print( line )


for coef, intercept, color in zip( svc.coef_,
            svc.intercept_, mglearn.cm3.colors):
    plt.plot( line,
        -(line * coef[0] + intercept)/coef[1],
        c=color)
        #x축, y축, 색상정보
    #x축, y축 크기 제한
    plt.ylim(-15, 15)
    plt.xlim(-10, 8)
    #제목줄
    plt.xlabel("특성0")
    plt.ylabel("특성1")
    plt.legend( ["클래스0", "클래서1", "클래스2",
    "클래스0경계", "클래스1경계", "클래스2경계"],
    loc=(1.01, 0.3))

# 경계 채우기
mglearn.plots.plot_2d_classification(svc, X,
    fill=True, alpha=.3)
    #alpha 값 1이면 불투명 0 완전투명
plt.show()

# 1.분류 객체 만들고
# 2. fit
# 3. predict