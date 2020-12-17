import numpy as np

X=np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)
X=X.flatten()
# x를 1차원 배열로 변환(평탄화)
print(X)
print(X[np.array([0,2,4])])
# 인덱스가 0,2,4인 원소 얻기
print(X)
# 원래 X는 유지된다.