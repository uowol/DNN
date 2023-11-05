# DNN
ToyProject-DNN

> 🤖: DNN은 "Deep Neural Network"의 줄임말로, 깊은 신경망 네트워크를 의미합니다. 이는 인공 신경망의 한 종류로, 여러 개의 은닉층을 가진 복잡한 구조를 갖고 있습니다. DNN은 기계 학습 및 딥 러닝 분야에서 사용되며, 이미지 및 음성 인식, 자연어 처리 등 다양한 작업에서 높은 성능을 보이는데 활용됩니다. DNN은 입력 데이터를 여러 층의 뉴런(노드)을 거쳐 처리하고 학습하여 복잡한 패턴을 인식하고 예측할 수 있습니다. 이러한 네트워크는 대용량의 데이터셋에서 학습하여 실제 세계의 복잡한 문제를 해결하는 데 사용됩니다.

# Todo
- [x] 문제 정의
- [ ] 데이터 수집
- [ ] 모델 구현
- [ ] 학습 및 결과 정리

# Problem definition
## Main Focus
- 하나의 층으로는 해결하지 못하는 문제
  - 가장 간단한 예로, XOR 문제의 해결이 있다.
- 공간적인 특징이나 순차적인 특징을 갖지 않는 데이터를 처리하는 문제
  - 시공간적으로 독립적이거나 추상화가 충분하게 이루어진 데이터에 대해 진행할 수 있다.
- Early Stopping Method의 사용
- 과소추정/과대추정의 중요도에 따른 Loss Function의 수정

## Todo List
- XOR 문제의 해결
- 유방암 데이터에 대한 악성 종양 분류 문제의 해결

# Data Collection
## XOR
|X1|X2|Y|
|:--:|:--:|:--:|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|0|

## 유방암 데이터
유방암 데이터는 sklearn에서 제공하는 데이터를 활용하였다.
```py
import pandas as pd 
from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()
df = pd.DataFrame(breast.data, columns=breast.feature_names)  # 독립변수 Xs
df['y'] = breast.target                                       # 종속변수 Y
```

```txt
RangeIndex: 569 entries, 0 to 568
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   mean radius              569 non-null    float64
 1   mean texture             569 non-null    float64
 2   mean perimeter           569 non-null    float64
 3   mean area                569 non-null    float64
 4   mean smoothness          569 non-null    float64
 5   mean compactness         569 non-null    float64
 6   mean concavity           569 non-null    float64
 7   mean concave points      569 non-null    float64
 8   mean symmetry            569 non-null    float64
 9   mean fractal dimension   569 non-null    float64
 10  radius error             569 non-null    float64
 11  texture error            569 non-null    float64
 12  perimeter error          569 non-null    float64
 13  area error               569 non-null    float64
 14  smoothness error         569 non-null    float64
 15  compactness error        569 non-null    float64
 16  concavity error          569 non-null    float64
 17  concave points error     569 non-null    float64
 18  symmetry error           569 non-null    float64
 19  fractal dimension error  569 non-null    float64
 20  worst radius             569 non-null    float64
 21  worst texture            569 non-null    float64
 22  worst perimeter          569 non-null    float64
 23  worst area               569 non-null    float64
 24  worst smoothness         569 non-null    float64
 25  worst compactness        569 non-null    float64
 26  worst concavity          569 non-null    float64
 27  worst concave points     569 non-null    float64
 28  worst symmetry           569 non-null    float64
 29  worst fractal dimension  569 non-null    float64
 30  y                        569 non-null    int32  
dtypes: float64(30), int32(1)
memory usage: 135.7 KB
```

# Modeling
## XOR
### Main Focus
- batch_size, epoch, learning_rate 하이퍼 파라미터에 따른 변화를 보이기
  - 가중치 시각화
  - 성능 그래프

## 악성 종양의 이진 분류 모델
### Main Focus
- 30개의 많은 변수를 어떻게 활용할 것인가
  - 다중공선성 확인
  - 차원축소

# Inference
## Train/Test Result

## Result
