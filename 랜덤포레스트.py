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
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)

print("train {}".format(forest.score(X_train, y_train)))
print("test {}".format(forest.score(X_test, y_test)))

print(forest.feature_importances_)


#차트그리기 중요!
def treeChart(feature_names, feature_importances):
    #특성의 개수
    n_feature = len(feature_importances)
    #수평 막대그래프
    plt.barh(np.arange(n_feature), #특성개수만큼 정수배열
        feature_importances,
        align="center" )
    #y축 눈금 제목
    plt.yticks(np.arange(n_feature), feature_names)
    plt.ylim(-1, n_feature)
    plt.show()

treeChart(np.array(cancer["feature_names"]),
        forest.feature_importances_)
