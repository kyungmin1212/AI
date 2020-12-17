import numpy as np

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp <=0:
        return 0
    else:
        return 1
"""
 b가 -0.1 이면 각 입력신호에 가중치를 곱한 값들의 합이 0.1을 초과할 때만 뉴런이 활성화
 b가 -20.0 이면 20을 넘지않으면 뉴런이 활성화되지 않는다.
 편향이란 용어는 한쪽으로 치우쳐 균형을 깬다는 의미
 편향의 값은 뉴련이 얼마나 쉽게 활성화되는지를 결정한다.
"""

def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))

