#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

#합격 불합격

#    [[]]   ,  []
# 공부사간 합격여부
"""
      40       PASS
      35       PASS
      20       FAIL
      15       FAIL
      18       FAIL
      19       PASS
      27       PASS
      26       PASS
      29       PASS
      30       FAIL
      31       PASS
      32       PASS
      33       FAIL
      42       PASS
"""
# 1단계 : 데이터 준비
pass_dataset = {
                "data" : np.array([[40],[35],[20],[15],[18],[19],[27],[26],[29],[30],[31],[32],[33],[34]]),
                "target" : np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]),
                "target_names" : np.array(['PASS','FAIL']),
                "feature_names" : ['time', 'result']
                }


#2단계 : 데이터를 훈련셋과 테스트셋으로 분리한다
#           7:3, 6:4 또는 8:2
from sklearn.model_selection import train_test_split

#train_test_split - 데이터 쪼개는 함수
X_train, X_test, y_train, y_test = train_test_split(
    pass_dataset["data"], pass_dataset["target"], random_state=0)#랜덤 값을 고정
            #입력데이터, 출력데이터, 데이터를 섞을지말지

#디폴트로 하면 75%대 25%로 쪼개진다
#3단계 학습을 한다.(머신러닝을 이용한다.)
#머신러닝 알고리즘중에 k이웃 분류 알고리즘을 사용한다
#KNeighborsClassifier 객체를 만들고 이 객체에
#이웃숫자를 매개변수로 전달한다. 이웃의 숫자에 따라서
#예측률이 달라진다. 보통 2~5정도를 많이 쓰는데
#적다고, 많다고 좋은게 아니라 다 테스트 해보고 적당한 것을
#선택해야한다

from sklearn.neighbors import KNeighborsClassifier
# knn 객체를 만든다
knn = KNeighborsClassifier(n_neighbors=1) #이웃을 하나로 놓고
# 학습시키기
knn.fit(X_train, y_train) #입력데이터, 출력데이터를 주고 학습을 시키면
# knn 객체 내부에 학습된 알고리즘이 저장되어 있다

X_new = np.array([[36]])
y_new = knn.predict(X_new)
print("36 시간 공부하면 합격? : ", pass_dataset["target_names"][y_new])

new_test = knn.predict(X_test)
print(knn.score(X_test, y_test))
print(X_test.shape)

"""
print("예측값 : ", new_test)
print("정답 : ", y_test)

#평가하기
score = knn.score( X_test, y_test)
print("예측률 : {}".format(score))
"""


