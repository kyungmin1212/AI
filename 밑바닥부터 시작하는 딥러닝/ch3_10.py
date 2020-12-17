import numpy as np
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network =pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))

# x,t=get_data()
# network=init_network()
# # 가중치와 편향 초기값
#
# accuracy_cnt=0
# for i in range(len(x)): # len(x) 60000개
#     y=predict(network,x[i]) # x 데이터 한개 784개의 입력값을 넣어준다.
#     p=np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
#     if p==t[i]: # y데이터 한개(t[i]) 와 확률이 가장높은 원소의 값이 같다면 정확도를 1을 올린다.( 이미 학습이 다되어있다)
#         accuracy_cnt+=1
#
# print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

x,t=get_data()
network=init_network()
# 가중치와 편향 초기값
batch_size=100
accuracy_cnt=0
for i in range(0,len(x),batch_size): # len(x) 60000개
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch) # x 데이터 batch사이즈만큼의 행과 784개의열의 입력값을 넣어준다.
    p=np.argmax(y_batch,axis=1) # 확률이 가장 높은 원소의 인덱스를 batch사이즈만큼의 행으로 얻게된다.
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])

    # y데이터 batch사이즈만큼데이터(t[i:i+batch_size])각각 행과 확률이 가장높은 원소의 값이 같다면 True를 리스트에 저장 아니면 False를 저장
    # print(p==t[i:i+batch_size])
    # p==t[i:i+batch_size] batch 사이즈 만큼 리스트에 각각 넘파이 배열끼리 비교하여 True/False로 구성된 bool 배열을 만단다.
    # np.sum(list)에서 list가 True와 False 로 이루어져있다면 True 값만큼 다 더하게 된다.

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))