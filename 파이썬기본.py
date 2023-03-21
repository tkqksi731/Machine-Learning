a = [1,2,3]
b = ["A", "B", "C"]
# zip : 두개이상의 리스트를 받아서 각각의 요소를
# tuple로 만들어서 반환한다
c = ["one", "two", "three"]
for i in zip(a, b, c):
    print(i)

for i in range(0, len(a)):
    print( a[i], b[i], c[i] )