import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

X=np.array([1.0,0.5])  # 1x2
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])  # 2x3
B1=np.array([0.1,0.2,0.3]) # 1x3

print(W1.shape)
print(X.shape)
print(B1.shape)

A1=np.dot(X,W1)+B1
print(A1)

Z1=sigmoid(A1)
print(Z1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

A2=np.dot(Z1,W2)+B2
print(A2)

Z2=sigmoid(A2)
print(Z2)

def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
print(A3)
Y=identity_function(A3)
print(Y)
# 항등 함수 identity_function()은 입력을 그대로 출력하는 합수이다.
# 굳이 정의할 필요는 없지만 그동안의 흐름과 통일하기 위해 이렇게 구현했다.
