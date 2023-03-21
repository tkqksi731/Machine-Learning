import pandas as pd

data = {
    "name":["이용희", "박동석", "김민수", "이태건", "남호석"],
    "kor":[90, 40, 50, 80, 70],
    "eng":[80, 60, 60, 70, 60],
    "mat":[70, 30, 40, 60, 70]
}

df = pd.DataFrame(data)
print( df.describe() ) # 데이터프레임 요약설명

# total을 추가해서 세과목 합계를 넣고자 한다

df["total"] = df.kor + df.eng + df.mat
print(df)
df["avg"] = df.total/3
print(df)
# print( df.name )
# print( df.kor )
# print( df.eng )
# print( df.mat )

# 데이터 3명분 추가
# 양아영 90, 90, 90
# 정자은 95, 85, 75
# 홍석원 70, 70, 80

df = df.append({"name":"양아영", "kor":90, "eng":90, "mat":90}, ignore_index=True)
df = df.append({"name":"정자은", "kor":95, "eng":85, "mat":75}, ignore_index=True)
df = df.append({"name":"홍석원", "kor":70, "eng":70, "mat":80}, ignore_index=True)

print(df)

df["total"] = df.kor + df.eng + df.mat
print(df)
df["avg"] = df.total/3
print(df)

#파일에 저장하기 - csv
df.to_csv("성적.csv", mode="w", encoding="cp949")

#또 읽기
df2 = pd.read_csv("성적.csv", encoding="cp949", index_col=0)
# index_col=0이 옵션 안넣어주면 파일 다시 읽을때 인덱스 또 만든다
print(df2)

#데이터 없는데 처음 데이터 프레임 만들때
column = ["name", "age", "rank"]
data = pd.DataFrame( columns=column )
data = data.append({"name":"홍길동", "age":40, "rank":1},
    ignore_index=True)
print(data)