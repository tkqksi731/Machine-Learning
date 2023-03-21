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
# X, y = mglearn.datasets.make_wave(n_samples=60)
cancer = datasets.load_breast_cancer()
# 2.데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    cancer["data"], cancer["target"],stratify=cancer["target"], random_state=42
)

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit( X_train, y_train )

#예측하기
y_pred = logistic.predict(X_test)
print(y_pred)
print(y_test)