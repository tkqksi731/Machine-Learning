import pandas as pd
import numpy as np

data = {
        'one': [1, 2, 3, 4, 5],
        'two': [6, 7, 8, 9, 10],
}

# pd.DataFame 이 dict 타입을 매개변수로 받아서 dataframe객체로
# 만든다. dataframe의 엑셀의 시트와 비슷한 역할을 한다
# 열이음, 인덱스가 구성된다.
# 특별히 인덱스를 지정하지 않으면 0,1,2,3,4...로 부여 된다
df = pd.DataFrame(data, index=["a", "b", "c", "d", "e"])
# 첫번째 인자로 dict타입, 두번째 인자로 index 부여함
# 인덱스는 부여 안할 수 있음
print(df)

# 데이터 접근하기
# iloc - 정수형태, 전통적인 인덱스를 이용한 방법
# df[1,2]

# #안되는 이유 : 판다스 만든사람은 파이썬 언어 만든 사람이 아니다
# print( df.iloc[1,2]) -------- XXXXXXX
# print( df.iloc[1][2]) ---------- XXXXXXX

#각 요소들에 접근할 수 있는 별도의 함수를 제공한다.
# lioc(i:Integer), ioc(열이름, 행이름),

print("0, 0번 데이터 : ", df.iloc[0, 0])
print("0, 1 데이터 : ", df.iloc[0, 1])

print("1, 0번 데이터 : ", df.iloc[1, 0])
print("1, 1 데이터 : ", df.iloc[1, 1])

print("1번행 : ", df.iloc[1, ])
print("1번행 : ", df.iloc[1, 1])

#열 추가하기
df["three"] = [10,20,30,40,50]
print(df)

df["four"] = np.arange(20, 25)
print(df)

df = df.append({"one":-1, "two":-2, "three":-3, "four":-4}, ignore_index=True)
print(df)

# iloc 이용해서 접근해보기
print( df.iloc[2:4, 2:4] )

# 2번 열만 출력하기
print( df.iloc[:, 2])

# 컬럼이름으로 출력하기
print( df.loc[0, "one"])
print( df.loc[:, "one":"three"])