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
data = pd.read_csv("boston_house_prices.csv",
                  header=1)
# print(data)

# 맨마지막의 타겟만 빼고 나머지 값들만
# dataframe -> ndarray 로 바꿔준다, axis=1 :열
X = data.drop(["MEDV"], axis=1).values
y = data["MEDV"].values

# 2.데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(
    X, y , random_state=42
)

# 3.결정트리
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

model.fit( X_train, y_train )
# 각 특성의중요도만 출력하기
print( model.feature_importances_ )