import numpy as np

X=np.array([1,2]) # 1x2
W=np.array([[1,3,5],[2,4,6]]) # 2x3
Y=np.dot(X,W) # 1x3
# 신경망에서 행렬의 곱을 할때 X가 앞에 있음을 주의하자 !
print(Y)