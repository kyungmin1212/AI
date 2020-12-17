import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=True,one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
# np.random.choice(60000,10) 이라 하면 0이상 60000미만 수중에서 무작위로 10개를 골라내는 것이다.
# train_size 60000개 중에 10개를 무작위로 골라낸것(10개 사진 골라내기 255 개 데이터로 표현된 사진 10장)(one_hot encoding 으로 표현된 결과값 10개)
print(batch_mask)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]
print(x_batch)
print(t_batch)