import pandas as pd

a = [1, 2, 3, 4, 5]
series1 = pd.Series(a)
print(series1, type(series1))

print("인덱스 : ", series1.index)
print("값: ", series1.values)

#index를 이용해 접근한다
print("0번째 데이터 : ", series1[0])
print("1번째 데이터 : ", series1[1])
print("2번째 데이터 : ", series1[2])
print("3번째 데이터 : ", series1[3])
print("4번째 데이터 : ", series1[4])
