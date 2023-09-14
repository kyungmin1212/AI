# XGBoost
- `pip install xgboost`
- `from xgboost import XGBRegressor`
    - 회귀
- `from xgboost import XGBClassifier`
    - 분류

## Table of Contents
- [Concepts](#concepts)
- [Code](#code)
--- 
### Concepts
- Gradient Boosting을 크게 업그레이드한 모델
    - eXtreme Gradient Boosting 의 줄임말
- **누락된 값 처리 가능(missing 매개 변수)**
    <details>
    <summary>설명</summary>

    - 일반적인 데이터 전처리 단계에서는 누락된 값을 평균값, 중앙값, 최빈값 등으로 채워주는 방법이 일반적입니다. 그러나 XGBoost와 같은 일부 고급 모델은 누락된 데이터를 처리하는 더욱 똑똑한 방법을 사용합니다.
    - XGBoost에서는 누락된 값이 있는 특성에 대한 분할 시, 누락된 값을 각각의 분할 방향(왼쪽 또는 오른쪽 자식 노드)으로 보내보고, 어느 쪽이 더 좋은 결과를 가져오는지를 결정합니다. 이러한 방식으로, 누락된 값이 있는 데이터도 모델 학습에 유용하게 사용됩니다. 이 과정은 모델 학습 중에 최적의 트리 구조를 찾기 위해 자동으로 수행됩니다.  
    - 이렇게 함으로써, XGBoost는 누락된 데이터를 직접적으로 처리하고, 누락된 데이터가 있어도 모델의 성능을 향상시킬 수 있는 방법을 찾습니다. 따라서 XGBoost를 사용할 때는 누락된 값을 미리 채울 필요가 없으며, 누락된 값 자체가 중요한 정보가 될 수 있습니다.
    </details>

    <details>
    <summary>예시 코드 1</summary>
    
    ```python
    import xgboost as xgb
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # 데이터 로딩
    boston = load_boston()
    X = boston.data
    y = boston.target

    # 누락된 값 추가 (예시를 위해 일부러 누락된 값 추가)
    X[10, 2] = np.nan
    X[20, 3] = np.nan

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DMatrix 생성 (XGBoost의 고유 데이터 구조)
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

    # 파라미터 설정
    param = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'reg:squarederror'
    }

    # 훈련
    bst = xgb.train(param, dtrain, num_boost_round=10)

    # 예측
    preds = bst.predict(dtest)

    # 첫 5개의 예측 출력
    print(preds[:5])

    ```

    </details>
- **속도 향상**
    <details>
    <summary>설명</summary>
    - 근사 분할 탐색 알고리즘
    - 희소성 고려 분할 탐색
    - 병렬 컴퓨팅
    - 캐시 고려 접근
    - 블록 압축과 샤딩
    </details>
- **정확도 향상**
    <details>
    <summary>설명</summary>
    - 자체적으로 규제를 추가하여 Gradient Boosting 이상으로 정확도를 높임.
    - 규제(regularization)는 분산을 줄이고 과대적합을 방지하기 위한 방법.
    - XGBoost는 Gradient Boosting과 Random Forest와 달리 학습하려는 목적 함수의 일부로 규제를 포함하고 있음.
    - 즉, XGBoost는 Gradinet Boosting의 규제 버전.
    </details>
    <details>
    <summary>예시 코드 1</summary>
    
    ```python
    import xgboost as xgb
    from xgboost import DMatrix
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # 데이터 로딩
    boston = load_boston()
    X = boston.data
    y = boston.target

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DMatrix 형식으로 데이터 변환
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # 파라미터 설정
    params = {
        'max_depth': 3,          # 트리의 최대 깊이
        'eta': 0.1,              # 학습률
        'objective': 'reg:squarederror',  # 회귀 문제
        'reg_alpha': 0.1,        # L1 규제 항 (alpha)
        'reg_lambda': 1.0,       # L2 규제 항 (lambda)
    }

    # 모델 학습
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # 예측
    predictions = bst.predict(dtest)

    # 첫 5개 예측값 출력
    print(predictions[:5])

    ```

    </details>

#### References
- [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://www.yes24.com/Product/Goods/108708980)
---

### Code

#### References
- [XGBoost와 사이킷런을 활용한 그레이디언트 부스팅](https://www.yes24.com/Product/Goods/108708980)
---