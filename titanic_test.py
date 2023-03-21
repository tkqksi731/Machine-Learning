import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import mglearn
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from keras import models 
from keras import layers

titanic_train = pd.read_csv("titanic/train_data.csv")
titanic_test = pd.read_csv("titanic/test_data.csv")

train_target = titanic_train['Survived']
train_data = titanic_train.drop(['Survived'], axis=1)

test_target = titanic_test['Survived']
test_data = titanic_test.drop(['Survived'], axis=1)


print(train_target.shape)
print(train_data.shape)

print(test_data[:10])

def KNei(train_data, train_target, test_data, test_target):
    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(train_data, train_target)

    #학습성과평가
    print("----KNeighborsClassifier-----")
    print("훈련세트 : {}".format(clf.score(train_data, train_target)))
    print("테스트 : {}".format(clf.score(test_data, test_target)))
    
    result = clf.score(test_data, test_target)
    return result

def Logist(train_data, train_target, test_data, test_target):
    log_reg = LogisticRegression()
    log_reg.fit(train_data, train_target)

    print("----LogisticRegression-----")
    print("훈련세트 : {}".format(log_reg.score(train_data, train_target)))
    print("테스트 : {}".format(log_reg.score(test_data, test_target)))

    result = log_reg.score(test_data, test_target)
    return result

def SVCtest(train_data, train_target, test_data, test_target):
    svctest = SVC()
    svctest.fit(train_data, train_target)

    print("----SVCtest-----")
    print("훈련세트 : {}".format(svctest.score(train_data, train_target)))
    print("테스트 : {}".format(svctest.score(test_data, test_target)))

    result = svctest.score(test_data, test_target)
    return result

def lanforest(train_data, train_target, test_data, test_target):
    forest = RandomForestClassifier(n_estimators=100, n_jobs =-1, random_state=0)
    #n_jobs =-1 시스템 내부 코어 최대로
    forest.fit(train_data, train_target)

    print("----lanforest-----")
    print("훈련세트 : {}".format(forest.score(train_data, train_target)))
    print("테스트 : {}".format(forest.score(test_data, test_target)))

    result = forest.score(test_data, test_target)
    return result

def DecisionTree(train_data, train_target, test_data, test_target):
    tree = DecisionTreeClassifier()
    tree.fit(train_data, train_target)

    print("----DecisionTree-----")
    print("train : {}".format(tree.score(train_data, train_target)))
    print("test : {}".format(tree.score(test_data, test_target)))

    result  = tree.score(test_data, test_target)
    return result

def xgboot(train_data, train_target, test_data, test_target):
    xgboot= xgb.XGBClassifier(random_state=0 , n_estimators=400 , max_depth=5 , learning_late=0.01)
    xgboot.fit(train_data, train_target)

    print("----xgboost-----")
    print("train : {}".format(xgboot.score(train_data, train_target)))
    print("test : {}".format(xgboot.score(test_data, test_target)))

    result  = xgboot.score(test_data, test_target)
    return result


def deeptest(train_data, train_target, test_data, test_target):
    #네트워크 모델 만들기
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu',input_shape=(14,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    #중간에 새로운 층 삽입
    #출력 layer 만들기
    model.add(layers.Dense(1, activation="sigmoid"))

    #모델 컴파일
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])

    #학습 시작하기
    model.fit(train_data, train_target, epochs=100, batch_size=30, validation_data=(test_data, test_target))

    #fit 함수가 학습에 필요했던 모든 역사(history)를 저장했다가 반환함
    results = model.evaluate(train_data, train_target)
    print(results)
    return results[1]

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

datalist.append(KNei(train_data, train_target, test_data, test_target))
datalist.append(Logist(train_data, train_target, test_data, test_target))
datalist.append(SVCtest(train_data, train_target, test_data, test_target))
datalist.append(lanforest(train_data, train_target, test_data, test_target))
datalist.append(DecisionTree(train_data, train_target, test_data, test_target))
datalist.append(xgboot(train_data, train_target, test_data, test_target))
datalist.append(deeptest(train_data, train_target, test_data, test_target))

treechart(namelist,datalist)