# import sys,os
# sys.path.append(os.pardir)
# 부모 디렉터의 파일을 가져올수 있는 방법 dataset 디렉토리가 현재 디렉토리 안에 있는게 아니라
# 부모 디렉토리에 있으면은 위와 같은 코드로 부모 디렉토리의 파일을 가져올 수 있도록 설정한다.
from dataset.mnist import load_mnist
# dataset 폴더에 있는 mnist 모듈로 부터 load_mnist 를 사용하겠다.
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False,one_hot_label=False)
# flatten 이 True 면 1x28x28 의 3차원 배열을 784개의 원소로 이뤄진 1차원 배열로 저장한다. False 면 1x28x28인 3차원 배열
# normalize 는 0~1 사이 값으로 정규화할지를 정한다. True 면 0~1 값으로 바꿔준다 . False 면 0~255 값 유지
# one_hot_label 이 True 면 숫자를 0과 1로 표현된 배열로 바꿔준다. 7-->[0,0,0,0,0,0,0,1,0,0] False 면 7 그대로 나온다.
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)