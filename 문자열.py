def typeConversion(nums):
    resultList = list()
    for num in nums:
        resultList.append(int(num))

    return resultList

file = open("data.txt", "r")
lines = file.readlines()
file.close()

numList = ()
for line in lines:
    s = line.replace("\n", "") # 줄바꿈 기호를 공백으로
    # print(s, type(s))
    nums = s.split(",") # 데이터를 , 를 쪼갠다
    # print( nums )
    nums = typeConversion(nums)
    print(nums)