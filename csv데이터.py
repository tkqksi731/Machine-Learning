import pandas as pd
import numpy as np

#데이터 읽는 순간 dataframe이다
data = pd.read_csv("./boston_house_prices.csv","r",
                  header=1)
# print(data)

# 맨마지막의 타겟만 빼고 나머지 값들만
# dataframe -> ndarray 로 바꿔준다, axis=1 :열
X = data.drop(["MEDV"], axis=1).values
y = data["MEDV"].values

print(X)
print(y)

# csv 파일 읽어서 데이터로 전환