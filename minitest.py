from konlpy.tag import Kkma
import pandas as pd

kk = Kkma()
sentimental_info = pd.read_csv('c:/miniproject/lexicon/polarity.csv'
                                , encoding='utf-8'
                                , index_col=False)

# 마지막 문장 제거
s = "수익성 확 떨어진 韓 대표기업시총 5위권 내 평균 영업이익률韓 7.4%…선진국은 모두 20%대업황 둔화·노조 리스크가 발목한국뿐 아니라 미국 일본 유럽 등 주요국 상장사의 2분기 실적시즌이 한창이다. 2분기 한국 상장사들의 영업이익이 급감하면서 수익성(영업이익률) 측면에서 글로벌 ‘간판’ 기업들에 비해 한참 뒤처진 것으로 나타났다.29일 금융감독원, 한국거래소에 따르면 지난 26일 기준 유가증권시장 시가총액 상위 다섯 곳(삼성전자·SK하이닉스·현대자동차·셀트리온·LG화학) 중 아직 실적을 발표하지 않은 셀트리온을 제외한 나머지 네 곳의 평균 영업이익률은 7.4%로 집계됐다. 이는 전년 동기(23.2%)보다 15.8%포인트 감소한 수치다. 미국(S&P500) 일본(닛케이225) 유럽(유로스톡스50)의 시총 5위권 내 종목의 올해 2분기 평균 영업이익률은 각각 28.6%, 23.2%, 21.6%로, 모두 20%대를 웃돌았다.동종 업종 내 기업 간 영업이익률을 비교해보면 한국 정보기술(IT) 업종 대표인 삼성전자와 SK하이닉스는 미국의 시총 상위 IT주들에 비해 수익성이 크게 뒤떨어졌다. 삼성전자와 SK하이닉스의 2분기 영업이익률은 각각 11.6%와 9.8%로, 1년 전에 비해 각각 13.8%포인트, 43.9%포인트 떨어졌다. 반면 미국의 시총 1~5위 기업 가운데 IT 업종에 속한 마이크로소프트(37.3%) 애플(21.2%) 페이스북(56.8%) 알파벳(22.5%) 등은 모두 높은 영업이익률을 나타냈다.한국 10대 기업 중 유일하게 2분기 실적이 대폭 개선된 현대차도 영업이익률은 일본 도요타의 절반 수준에 머물렀다. 현대차는 2분기에 전년 동기보다 30.1% 증가한 1조2377억원의 영업이익을 올렸다. 금융정보업체 에프앤가이드가 집계한 실적 발표 전 시장 컨센서스(증권사 추정치 평균·1조649억원)를 7.0% 웃도는 ‘서프라이즈’ 수준의 성적이다. 하지만 영업이익률은 4.5%에 그쳤다. 도요타의 2분기 영업이익률은 8.6%였다.전문가들은 반도체 화학 등 한국 경제를 이끄는 주력 업종이 모두 심각한 업황 둔화에 빠진 가운데 고용경직성 심화로 적극적인 구조조정에 나서기 어려운 게 한국 기업들의 수익성을 급격히 악화시키고 있다고 지적했다. 이만우 고려대 경영학과 교수는 “경기불황기에는 생산량을 적극적으로 줄여 재고 자산을 감소시키는 등의 노력이 필요한데 한국은 노동조합의 입김이 세 생산을 줄이지 못하면서 부담이 커지고 있다”고 설명했다.송종현 기자 scream@hankyung.com▶ 네이버에서 한국경제 뉴스를 받아보세요▶ 한경닷컴 바로가기  ▶ 모바일한경 구독신청 ⓒ 한국경제 & hankyung.com, 무단전재 및 재배포 금지"
a = kk.sentences(s)  # 문장별 구문
a.pop(-1)
s = "".join(a)
# 마지막 문장 제거 후 다시 문자열로 변경

pos = kk.pos(s)
n_pos = []
for p in pos:
    st = "".join([p[0], "/", p[1]])
    n_pos.append(st)

length = len(n_pos)
i = 0
k = 0
# 실제 검색 형태소
real_pos = []
# POS,NEG,COMP 입력
sent_list = []
while i < length:
    val_list = []
    for ngram in ngram_list:
        n_split = ngram.split(';')[0]
        if n_pos[i] == n_split:
            val_list.extend(sentimental_info[ngram == sentimental_info['ngram']]['ngram'].values)

    val_len = len(val_list)     
    if val_len == 0:
        i = i + 1
    elif val_len == 1:
        ngram = sentimental_info[n_pos[i] == sentimental_info['ngram']]['ngram'].values
        value = sentimental_info[n_pos[i] == sentimental_info['ngram']]['max.value'].values
        prop = sentimental_info[n_pos[i] == sentimental_info['ngram']]['max.prop'].values
        print("형태소 : {}, value : {}, porp : {}".format(ngram, value, prop))
        real_pos.extend(n_pos[i])
        sent_list.extend(value)
        i = i + 1
    else:
        key = n_pos[i]
        tmp_key = ""
        while key in val_list:
            if i < len(n_pos) :
                k = i + 1
                tmp_key = key
                key = "".join([n_pos[i], ";", n_pos[k]])
                i = k
            else:
                tmp_key = key
                # print(sentimental_info[key == sentimental_info['ngram']]['ngram'].values)
                i = i + 1
        else:
            if tmp_key != "":
                real_pos.append(tmp_key)
            i = k
            k = 0
            ngram = sentimental_info[tmp_key == sentimental_info['ngram']]['ngram'].values
            value = sentimental_info[tmp_key == sentimental_info['ngram']]['max.value'].values
            prop = sentimental_info[tmp_key == sentimental_info['ngram']]['max.prop'].values
            print("형태소 : {}, value : {}, porp : {}".format(ngram, value, prop))
            sent_list.extend(value)
            
print(sent_list.count("POS"))
print(sent_list.count("NEG"))