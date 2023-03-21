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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

a = [GaussianNB(), BernoulliNB(), LogisticRegression(), LinearSVC()]
name = ["GaussianNB", "BernoulliNB", "LogisticRegression", "LinearSV"]

#객체리스트

# 다항분포
for item in range(0, len(a)):
    nb1 = a[item]
    nb1.fit( X_train, y_train )
    print(" == {} == ".format(name[item]))
    print("훈련성과 {0}".format(nb1.score(X_train, y_train)))
    print("테스트 성과 {0}".format(nb1.score(X_test, y_test)))