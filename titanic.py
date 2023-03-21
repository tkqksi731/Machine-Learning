import pandas as pd
import numpy as np
from sklearn import datasets
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from keras import models
from keras import layers


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

train = pd.read_csv('titanic/train_data.csv')
test = pd.read_csv('titanic/test_data.csv')


X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]


X_test = test.drop("Survived", axis=1)
y_test = test["Survived"]

#########################################

print("로지스틱")
def Logist(X_train, X_test, y_train, y_test):
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # for i in range(0, len(y_pred)):
    #     print(y_pred[i], y_test[i], y_pred[i]==y_test[i])

    print("훈련셋 {}".format(model.score(X_train, y_train)))
    print("테스트 {}".format(model.score(X_test, y_test)))
    result = model.score(X_test, y_test)
    return result



print("k-nn 이웃분류")
def KNei(X_train, X_test, y_train, y_test):

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    print("훈련세트 {:.2f}".format(clf.score(X_train, y_train)))
    print("테스트세트 {:.2f}".format(clf.score(X_test, y_test)))

    result = clf.score(X_test, y_test)
    return result



print("서포트벡터머신")
def SVCtest(X_train, X_test, y_train, y_test):

    forest = SVC()
    forest.fit(X_train, y_train)

    print("train {}".format(forest.score(X_train, y_train)))
    print("test {}".format(forest.score(X_test, y_test)))

    result = forest.score(X_test, y_test)
    return result



print("결정트리")
def DecisionTree(X_train, X_test, y_train, y_test):
    
    tree = DecisionTreeClassifier()
    tree.fit( X_train, y_train)

    print("train : {}".format( tree.score( X_train, y_train )))
    print("test : {}".format( tree.score( X_test, y_test )))

    result = tree.score(X_test, y_test)
    return result



print("랜덤포레스트")
def lanforest(X_train, X_test, y_train, y_test):

    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    forest.fit(X_train, y_train)

    print("train {}".format(forest.score(X_train, y_train)))
    print("test {}".format(forest.score(X_test, y_test)))

    # print(forest.feature_importances_)
    result = forest.score(X_test, y_test)
    return result


print("xgboost")
def xgboot(X_train, X_test, y_train, y_test):

    forest = xgb.XGBClassifier(random_state=0, n_estimators=400,
        max_depth=5, learning_rate=0.01)
    forest.fit(X_train, y_train)

    print("train {}".format(forest.score(X_train, y_train)))
    print("test {}".format(forest.score(X_test, y_test)))
    
    result = forest.score(X_test, y_test)
    return result


print("이진분류")
def deeptest(X_train, X_test, y_train, y_test):

    #네트워크 모델 만들기
    model = models.Sequential()

    # 입력 layer 만들기
    model.add( layers.Dense(16, activation="relu", input_shape=(14,)))
    #중간에 새로운 층 삽입
    model.add(layers.Dense(16, activation="relu"))
    #출력 layer 만들기, 2진분류의 경우 다르게 작성
    model.add(layers.Dense(1, activation="sigmoid"))

    #모델컴파일
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    #학습시작하기
    model.fit( X_train, y_train, epochs=100, batch_size=100, validation_data=(X_test, y_test))
    # fit함수가 학습에 필요했던 모든 역사(history)를 저장했다가 변환한다
    results = model.evaluate(X_train, y_train)
    print(results)
    return results[1]


import matplotlib.pyplot as plt

def treechart(feature_names, feature_importances):
    #특성의 개수
    n_feature = len(feature_importances)
    #수평 막대그래프,    #특성개수만큼 정수배열
    plt.barh(np.arange(n_feature), feature_importances, align="center") 
    #y축 눈금 제목
    plt.yticks(np.arange(n_feature), feature_names)
    plt.ylim(-1, n_feature)
    plt.show()

namelist=['KNei', 'Logist', 'SVCtest', 'lanforest', 'DecisionTree', 'xgboost', 'deeptest']
datalist = []

datalist.append(KNei(X_train, X_test, y_train, y_test))
datalist.append(Logist(X_train, X_test, y_train, y_test))
datalist.append(SVCtest(X_train, X_test, y_train, y_test))
datalist.append(lanforest(X_train, X_test, y_train, y_test))
datalist.append(DecisionTree(X_train, X_test, y_train, y_test))
datalist.append(xgboot(X_train, X_test, y_train, y_test))
datalist.append(deeptest(X_train, X_test, y_train, y_test))

treechart(namelist,datalist)