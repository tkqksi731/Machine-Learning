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
#X, y = mglearn.datasets.make_wave(n_samples=60)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# 2.데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    cancer["data"], cancer["target"], stratify=cancer["target"], random_state=42)

#계속해서 데이터를 쪼개나가면서 트리를 확장하는데
#냅두면 모든 데이터가 분리될때까지 트리
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit( X_train, y_train)

print("train : {}".format( tree.score( X_train, y_train )))
print("test : {}".format( tree.score( X_test, y_test )))

#트리의 깊이가 너무 깊어지는걸 막아보자
tree = DecisionTreeClassifier(max_depth=4)
tree.fit( X_train, y_train)

print("train : {}".format( tree.score( X_train, y_train )))
print("test : {}".format( tree.score( X_test, y_test )))

#특성의 중요도
print( tree.feature_importances_)

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
            tree.feature_importances_)
