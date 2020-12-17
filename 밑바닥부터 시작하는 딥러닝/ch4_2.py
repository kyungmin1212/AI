import numpy as np

def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))
# log안에 값이 0이 되면 무한대가 되어 더이상 계산을 할수 없기 때문에 -무한대를 방지하기 위하여 delta를 더해줬다.

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

print(cross_entropy_error(np.array(y),np.array(t)))
# 0.51082
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy_error(np.array(y),np.array(t)))
# 2.30258