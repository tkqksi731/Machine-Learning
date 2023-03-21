# dict -> dataframe로 바꾸기

import pandas as pd

# feature(특성): Name, Location, Age
data = {
    "Name":["홍길동", "임꺽정", "박동석", "이용희", "김민수"],
    "Location":["서울시", "부산시", "대구시", "광주시", "울산시"],
    "Age":[23, 49, 30, 27, 24]
}

data_pandas = pd.DataFrame(data)

print( data_pandas.keys() )
print( data_pandas )

print( data_pandas[ data_pandas.Age>=30 ])