import numpy as np

# np.logspace(시작,끝,개수,로그값의 간격(1->0.1->0.01->0.001-> ...)
x=np.logspace(0,11,12,base=0.1,dtype='float64')
# (1+x)^(1/x) 를 np.power 을 사용한것
# y=(1+x)**(1/x)
# print(y)
y=np.power(1+x,1/x)
for i in range(11):
    print(f'x={x[i]:.10f},y={y[i]:.10f}')