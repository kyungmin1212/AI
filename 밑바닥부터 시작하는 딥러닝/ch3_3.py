import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
# np로 해주는 이유 x 값에 배열이 들어와도 실행될수 있도록 하기 위함이다.

x=np.array([-1,1,2])
print(sigmoid(x))

x=np.arange(-5,5,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def relu(x):
    return np.maximum(0,x)