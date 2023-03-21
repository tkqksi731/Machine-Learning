import numpy as np
import pandas as pd

a = [1,2,3,4,5,6,7,8,9,10]

x = np.array(a)
#타입전환 list -> ndarray
#사이킷런 라이브러리는 무조건 ndarray타입이나 dataframe타입만
#사용한다 list -> ndarray로 바꿔서 써야 하고
# dict -> dataframe으로 바꿔서 써야 한다
print(type(a), a)
print(type(x), x)

import matplotlib.pyplot as plt
# 특정간격으로 데이터 생성하기
# linespace(시작, 종료, 구간갯수)로 나누어서 벡터를 생성한다

x = np.linspace(-10, 10, 100)
print(x)

# y = f(x)

y = np.sin(x) #벡터 받아가서 벡터 내놓음
# x 벡터에 맞ㅇ추어서 x의 개수 만큼 데이터 생성한다. 새로운 벡터
# y가 만들어진다

print( y )

plt.plot(x, y, marker="x")
plt.show() #jupyter 아닐때