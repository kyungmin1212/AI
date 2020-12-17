import numpy as np

# 정답이 2인 경우다 (one-hot encoding) (실제값 )
t=[0,0,1,0,0,0,0,0,0,0]

# 각 원소 추정값 2인경우가 0.6으로 가장 확률이 높다. (출력값 )
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

# 2일 확률이 가장 높다고 추정한 경우(0.6)
print(sum_squares_error(np.array(y),np.array(t)))
# sum_squares_error 값이 0.0975 로 상당히 낮은 값이 나왔다.

# 7일 확률이 가장 높다고 추정한 경우(0.6)
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squares_error(np.array(y),np.array(t)))
# sum_squares_error 값이 0.5975 로 높은 값이 나왔다.
