import numpy as np

A=np.array([[1,2],[3,4]])
print(A)
print(A.shape,A.dtype)

B=np.array([[3,0],[0,6]])
print(A+B)
print(A*B)
print(np.dot(A,B))
print(np.matmul(A,B))
# *는 요소끼리의 곱이다.
# dot 내적 , matmul 행렬의 곱
print(A*10)