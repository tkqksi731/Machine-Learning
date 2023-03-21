#html 문서 파싱용 라이브러리
from bs4 import BeautifulSoup
#html 문서를 읽어오는 라이브러리
from urllib.request import urlopen
import pandas as pd
import csv
import urllib

data = {
    "날짜":[],
    "종가":[],
    "시가":[],
    "고가":[],
    "저가":[],
    "거래량":[],
}

df = pd.DataFrame(data)

code = "034220"
wantPage = 3

html = urlopen("http://finance.naver.com/item/sise_day.nhn?code=" + code)
s = html.read()


bsObj = BeautifulSoup(s, "html.parser")
table = bsObj.find_all("table", {"align":"center"})
page = table[0].find_all("td", {"class":"pgRR"})

for page in range(1, wantPage+1):
    html = urlopen("http://finance.naver.com/item/sise_day.nhn?code=" + code + '&page=' + str(page))
    srlists=bsObj.find_all("tr")
    isCheckNone = None

    for i in range(1,len(srlists)-1):
        if(srlists[i].span != isCheckNone):
            srlists[i].td.text

            print(
                srlists[i].find_all("td",align="center")[0].text.replace(",","")
                , srlists[i].find_all("td",class_="num")[0].text.replace(",","")
                , srlists[i].find_all("td",class_="num")[2].text.replace(",","")
                , srlists[i].find_all("td",class_="num")[3].text.replace(",","")
                , srlists[i].find_all("td",class_="num")[4].text.replace(",","")
                , srlists[i].find_all("td",class_="num")[5].text.replace(",",""))

    
            df = df.append({
                "날짜": srlists[i].find_all("td",align="center")[0].text.replace(",","")
                ,"종가": srlists[i].find_all("td",class_="num")[0].text.replace(",","")
                ,"시가": srlists[i].find_all("td",class_="num")[2].text.replace(",","")
                ,"고가": srlists[i].find_all("td",class_="num")[3].text.replace(",","")
                ,"저가": srlists[i].find_all("td",class_="num")[4].text.replace(",","")
                ,"거래량": srlists[i].find_all("td",class_="num")[5].text.replace(",","")
                }, ignore_index=True)


df.to_csv("주식일거래량.csv", mode="w", encoding="cp949")