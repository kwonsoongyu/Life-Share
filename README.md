# Life-Share(Sharing Economic Platform)
2022년 제 2회 금융 데이터 활용 경진대회


<br>

## 🖥 프로젝트 소개
‘Life-Share'는 공유 경제 활성화로 사용자의 소비자들의 효율적인 소비문화를 유도하고 사회문제 해결에 이바지하기 위한 플랫폼이다.

<br>

## 👩🏻‍💻 멤버 구성

| [권순규](https://github.com/kwonsoongyu)    |   [김광모](https://github.com/kkm0406)    | [전승원](https://github.com/s2eung1)



<br>
 
## 👩🏻‍💻 프로젝트 개요
금융데이터와 공유 경제 데이터의 분석으로 사회 경제적 문제를 시각화하고 정의한 문제로부터 도출된 답과 공유 경제의 장점과 특성을 사용한다. 
이를 통해 소비자들의 효율적인 소비문화를 유도하고 사회문제 해결에 이바지할 수 있는 공유 경제 활성화 플랫폼 'Life-Share'를 제안한다.
<br>

## ⚒️ 데이터 분석 알고리즘
1. DBSCAN(Density-Based Spatial Clustering of Application with Noise)
군집간의 거리를 이용한 밀도 기반 클러스터링 기법으로 기준점으로부터의 거리인 epsilon과 해당 반경 이내의 있는 점의 수인 minsample값을 활용하여 
어떠한 군집에도 속하지 않는 이상치 값을 제거하여 데이터셋을 정제하는데 활용했다. 

2. Firebase. SVM(Support Vector Machine)
분류의 성능이 뛰어난 지도 학습 알고리즘으로 데이터가 어떤 카테고리에 속할지 판단하는 비확률적 이진 선형 모델을 생성 후 
train set과 test set을 활용하여 모델 성능 평가에 활용했다.

3. MLR(Multi Linear Regression)
예측 및 분류 방법 중 하나로 예측에 활용되는 모델로 종속변수와 예측변수 사이의 관계를 적합시키기 위해 사용한다. X는 결과값에 영향을 미치는 변수 Y는 
결과값이며 알고자하는 값이다.

4. LSTM(Long Short Term Memory)
기존의 RNN이 출력과 먼 위치에 있는 정보를 기억할 수 없다는 단점을 보완하여 장/단기 기억을 가능하게 
설계한 신경망 구조 알고리즘으로 시계열 처리를 통해 공유경제의 전망에 대한 예측에 활용했다.

5. KNN(K-neighbor-nearest)
지도 학습의 한 종류로 머신러닝에서 데이터를 가장 가까운 유사 속성에 따라 분류하여 라벨링하는 거리기반 분류분석 모델로 데이터로부터 거리가 가까운 
‘k’ : neighbor개의 데이터의 레이블을 참조하여 분류하는 알고리즘으로 거리 측정 시 ‘유클리디안 거리’ 계산법을 사용한다. 

6. Decision Tree, Random forest
분류와 회귀 모두 가능한 지도 학습 모델 중 하나로 특정 기준에 따라 데이터를 구분하는 알고리즘으로 한 번의 분기 때마다 변수 영역을 두 개로 구분한다. 
Random forest :Decision tree 모델 여러 개를 훈련시켜서 그 결과를 종합해 예측하는 앙상블 알고리즘으로 전체 trainset에서 중복을 허용한 dataset으로 개별 
decision tree를 훈련시키는 배깅 방식을 사용

7. Smote, Adasyn
Smote : 대표적인 오버 샘플링 기법 중 하나로 낮은 비유로 존재하는 클래스의 데이터를 KNN 알고리즘을 활용하여 비중을 새롭게 생성하는 방법
Adasyn : 오버 샘플링 기법 중 하나로 샘플링 개수를 데이터 위치에 따라 다르게 설정하고 이 때 가중치를 통해 SMOTE를 적용한다

### Tool & Library
- Tool : Vscode, Jupyter notebook
- Library : numpy, pandas, plotly
- Language : python


<br>

## 🌈 구현 기능

### 메인 페이지
![main]

### Fing Market, 지역 페이지
![market]

### 지역리스트, 페스티벌 상세 페이지
![fing1]

### 찜 페이지, 마이페이지
![fing2]

### Web
![web]

<br>

## 🏆 수상 내역

<p align="left">
  <img src="./img/poster.png" width="300px" height="400px">
  <img src="./img/award.png" width="300px" height="400px">
  <br>
</p>

<!-- Image Refernces -->
[조민수]: /img/%E1%84%8C%E1%85%A9%E1%84%86%E1%85%B5%E1%86%AB%E1%84%89%E1%85%AE.png
[김광모]: /img/%E1%84%80%E1%85%B5%E1%86%B7%E1%84%80%E1%85%AA%E1%86%BC%E1%84%86%E1%85%A9.png
[유병주]: /img/%E1%84%8B%E1%85%B2%E1%84%87%E1%85%A7%E1%86%BC%E1%84%8C%E1%85%AE.png
[정세연]: /img/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%89%E1%85%A6%E1%84%8B%E1%85%A7%E1%86%AB.png
[전승원]: /img/%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%89%E1%85%B3%E1%86%BC%E1%84%8B%E1%85%AF%E1%86%AB.png
[천은정]: /img/%E1%84%8E%E1%85%A5%E1%86%AB%E1%84%8B%E1%85%B3%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC.png
[fing1]: /img/fing-00.gif
[fing2]: /img/fing-11.gif
[web]: /img/fing-web.gif
[main]: /img/%E1%84%86%E1%85%A6%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%91%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%8C%E1%85%B5(1).gif
[market]: /img/%E1%84%86%E1%85%A1%E1%84%8F%E1%85%A6%E1%86%BA%2C%E1%84%8C%E1%85%B5%E1%84%8B%E1%85%A7%E1%86%A8(2).gif
[fing3]: /img/222.gif
[포스터]: /img/poster.png
[상장]: /img/award.png
