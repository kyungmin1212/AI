import numpy as np

def step_function(x):
    if x>0:
        return 1
    else:
        return 0
# 이 구현은 넘파이 배열을 인수로 넣을 수없다. 오로지 실수 만가능

def step_function1(x):
    y=x>0
    return y.astype(np.int)
# 배열을 인수로 넣을수 있다.
x=np.array([-1,1,2])
print(x)
y=x>0
print(y)
print(y.astype(np.int))
# 넘파이 배열의 자료형을 변환할 때는 astype() 메서드 사용
# bool 을 int 로 변환하면 True 는 1로 False 는 0으로 변환된다.

print(step_function1(x))