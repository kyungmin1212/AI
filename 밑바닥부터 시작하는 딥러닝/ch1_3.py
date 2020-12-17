import numpy as np

A=np.array([[1,2],[3,4]])
B=np.array([10,20])
print(A*B)
# B차원을 [10,20] --> A에 맞춰 [[10,20],[10,20]] 으로 바꿔준다.
# 브로드캐스트