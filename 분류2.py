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

cancer = datasets.load_breast_cancer()

#데이터 확인
#갖고 있는 키값 확인하기
#키값들 :  dict_keys(['data', 'target', 'target_names',
# 'DESCR', 'feature_names', 'filename'])
print("키값들 : ", cancer.keys())
data = cancer["data"]
print("데이터의 규모 : ", data.shape)
print( data[:5] ) # 앞에서 5개만 미리 확인

#특성들의 이름
print( cancer["feature_names"])
target = cancer["target"]
print(target[:5])

X_train, X_test, y_train, y_test = train_test_split(data, target,stratify=cancer.target, random_state=66)


train_scores = []
test_scores = []

for n in range(1, 11):
    clf1 = KNeighborsClassifier(n_neighbors=n)
    clf1.fit( X_train, y_train)

    print("이웃이 {}일때 -----".format(n))
    print("훈련셋 {:.2f}".format(clf1.score(X_train, y_train)))
    print("테스트셋 {:.2f}".format(clf1.score(X_test, y_test)))

    train_scores.append( clf1.score(X_train, y_train))
    test_scores.append( clf1.score(X_test, y_test))

print(train_scores)
print(test_scores)

#x축 - 이웃의 개수
#y축 - 평가값
#이웃의 개수
neighbors=list(range(1,11))
plt.plot(neighbors, train_scores) # 훈련데이터셋 평가
plt.plot(neighbors, test_scores) # 테스트데이터셋 평가
plt.show()