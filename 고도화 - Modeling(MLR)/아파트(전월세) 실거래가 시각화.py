import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
plt.rcParams['font.family']='Malgun Gothic'
df = pd.read_csv('아파트(전월세)_실거래가.csv')
#지역 주소를 시군구 단위로 그룹화하기 위한 전처리
df['시군구'] = df['시군구'].str.split(' ')
#지역별 그룹화를 위한 전처리
for i in range(len(df)):
if df['시군구'][i][0] == '강원도':
df['시군구'][i] = '강원'
if df['시군구'][i][0] == '경기도':
df['시군구'][i] = '경기'
if df['시군구'][i][0] == '경상남도':
df['시군구'][i] = '경남'
if df['시군구'][i][0] == '경상북도':
df['시군구'][i] = '경북'
if df['시군구'][i][0] == '광주광역시':
df['시군구'][i] = '광주'
if df['시군구'][i][0] == '대구광역시':
df['시군구'][i] = '대구'
if df['시군구'][i][0] == '대전광역시':
df['시군구'][i] = '대전'
if df['시군구'][i][0] == '부산광역시':
df['시군구'][i] = '부산'
if df['시군구'][i][0] == '서울특별시':
df['시군구'][i] = '서울'
if df['시군구'][i][0] == '세종특별자치시':
df['시군구'][i] = '세종'
if df['시군구'][i][0] == '울산광역시':
df['시군구'][i] = '울산'
if df['시군구'][i][0] == '인천광역시':
df['시군구'][i] = '인천'
if df['시군구'][i][0] == '전라남도':
df['시군구'][i] = '전남'
if df['시군구'][i][0] == '전라북도':
df['시군구'][i] = '전북'
if df['시군구'][i][0] == '제주특별자치도':
df['시군구'][i] = '제주'
if df['시군구'][i][0] == '충청남도':
df['시군구'][i] = '충남'
if df['시군구'][i][0] == '충청북도':
df['시군구'][i] = '충북'
# 보증금(만원), 월세(만원) column 형변환
df['보증금(만원)'] = df['보증금(만원)'].str.replace(",", "")
df['보증금(만원)'] = df['보증금(만원)'].astype('float')
df['월세(만원)'] = df['월세(만원)'].str.replace(",", "")
df['월세(만원)'] = df['월세(만원)'].astype('float')

#시도별 그룹화
data1 = df.groupby(by=['시군구']).mean()
#새로운 Dataframe 생성
data2 = data1.loc[['서울', '부산', '대구', '인천', '광주',
'대전', '울산', '세종', '경기', '강원',
'충북', '충남', '전북', '전남', '경북', '경남', '제주']]

df1 = pd.read_csv('대중교통및교육시설.csv')
df2 = pd.read_csv('방범시설및조경면적.csv')
df3 = pd.read_csv('지역별주차장.csv')
df2 = df2.rename(columns = {'지역(1)' : '지역', '2019' : '호수 (호)','2019.1' : 'CCTV','2019.2':'조경면적'})
df_data5 = df2.drop([0,1])
df_data5.reset_index(inplace=True)
#index column 제거
                            
df_result = df_data5.drop(['index'],axis='columns')
df3.columns=['지역(1)','호수','주차장 총대수','주차장 옥내','주차장 옥내(전기자동차)','주차장 옥외','주차장옥외(전기자동차)']
df3.drop([0,1,2],axis=0,inplace=True)
df3.reset_index(inplace=True)

#'지역(1)' , '호수' column 제거
df__1=df3.drop(['지역(1)','호수'],axis=1)

#df_result에 병합
df_result = pd.concat([df_result,df__1],axis=1)
df1.columns = ['지역','호수','대중교통 1km 이내,단지수','대중교통 1km 초과 단지수','교육시설 초등학교','교육시설 중고등학교','교육시설 대학교 이상']
df1.drop([0,1,2,3],axis=0,inplace=True)
df1.reset_index(inplace=True)

#'지역(1)' , '호수' column 제거
df_2 = df1.drop(['index','지역','호수'],axis=1)

#df_result에 병합
df_result = pd.concat([df_result,df_2],axis=1)
data2.reset_index(inplace=True)

#df_result에 병합
df_result = pd.concat([df_result, data2], axis=1)

#최종 df_result Dataframe 생성
df_result = df_result.drop(['index', '지역','호수 (호)','주차장 총대수','주차장 옥내(전기자동차)', '주차장 옥외(전기자동차)', '시군구'], axis = 1)
df_result['월세(만원)'].hist(bins=100) #종속변수 y를 정한 후 (예측대상인 y에 대한 정보 확인)
               
#이상치 탐지를 위한 Box plot 시각화
fig = px.box(df_result, y="월세(만원)")
fig.show()
               
#df_result에 scaling 적용
def standard_scaling(df1, scale_columns):
  for col in scale_columns:
    series_mean = df1[col].mean()
    series_std = df1[col].std()
    df1[col] = df_result[col].apply(lambda x : (x-series_mean)/series_std)
    return df1
               
df_result = df_result.astype(float)
scale_columns = ['CCTV', '조경면적', '주차장 옥내', '주차장 옥외', '대중교통 1km 이내,단지수',
'대중교통 1km 초과 단지수', '교육시설 초등학교', '교육시설 중고등학교', '교육시설 대학교 이상',
'전용면적(㎡)', '보증금(만원)', '월세(만원)', '건축년도']
df_df1 = standard_scaling(df_result, scale_columns)
               
#월세(만원)을 독립변수로 설정
df_df1=df_df1.rename(columns={'월세(만원)':'y'})

#학습 데이터와 테스트 데이터로 분리
X = df_df1[df_df1.columns.difference(['y'])]
y=df_df1['y']

#검증(test)는 20%로 진행 -> test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#회귀 분석 객체 생성(선형 회귀 모델 생성)
lr=linear_model.LinearRegression()

#fit()은 기울기와 절편을 전달하기 위함.
model = lr.fit(X_train, y_train)

# 학습된 계수를 출력
print(lr.coef_)
# 상수항을 출력
print(lr.intercept_)
      
#완성한 모델을 이용해 새로운 값을 가직 예측, test용으로 썼던 X_test사용
x_new=X_test
y_new=model.predict(x_new)
y_compare={'y_test':y_test, 'y_predicted':y_new}
df2=pd.DataFrame(y_compare)

#train set과 예측 데이터 시각화
plt.rcParams['font.family']='AppleGothic'
df2.plot(y=['y_test', 'y_predicted'], kind="bar")
               
# 회귀분석모델을 평가하고 성늘 높이기 R2 Score, RMSE Score 사용
print(model.score(X_train, y_train)) # train R2 score를 출력
print(model.score(X_test, y_test)) # test R2 score를 출력
               
y_predictions = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_predictions))) # train RMSE score를 출력
y_predictions = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_predictions))) # test RMSE score를 출력

plt.rcParams['font.family']='Malgun Gothic' #'Malgun Gothic'
df_df1_corr=df_df1[['CCTV', '조경면적', '주차장 옥내', '주차장 옥외', '대중교통 1km 이내,단지수',
'대중교통 1km 초과 단지수', '교육시설 초등학교', '교육시설 중고등학교', '교육시설 대학교 이상', '전용면적(㎡)',
'보증금(만원)', 'y', '건축년도']]
plt.rcParams['figure.figsize']=(15,10)
sb.heatmap(df_df1_corr.corr(),
        annot=True,
        cmap='Reds',
        vmin = -1, vmax = 1
)
               
# VIF를 통한 다중공선성 파악
# 피처마다의 VIF 계수를 출력합니다.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_df1_corr.values, i) for i in range(df_df1
_corr.shape[1])]
vif["features"] = df_df1_corr.columns
               
               
#VIF 계수가 높은 feature제거
df_df1_corr=df_df1_corr.drop(['교육시설 초등학교', '교육시설 중고등학교'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_df1_corr.values, i) for i in range(df_df1
_corr.shape[1])]
vif["features"] = df_df1_corr.columns              
              
#VIF 계수가 높은 feature제거
df_df1_corr=df_df1_corr.drop(['주차장 옥내','조경면적'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_df1_corr.values, i) for i in range(df_df1
_corr.shape[1])]
vif["features"] = df_df1_corr.columns
                           
#VIF 계수가 높은 feature제거
df_df1_corr=df_df1_corr.drop(['대중교통 1km 이내,단지수', '주차장 옥외'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_df1_corr.values, i) for i in range(df_df1
_corr.shape[1])]
vif["features"] = df_df1_corr.columns 
               
               
#VIF 계수가 높은 feature제거
df_df1_corr=df_df1_corr.drop(['보증금(만원)'], axis=1)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_df1_corr.values, i) for i in range(df_df1
_corr.shape[1])]
최종 결과물 (분석 코드) 105
vif["features"] = df_df1_corr.columns
               
df_df1_corr_re=df_df1_corr[['CCTV', '대중교통 1km 초과 단지수', '교육시설 대학교 이상', '전용면적(㎡)', 'y', '건축년도']]
X=df_df1_corr_re[df_df1_corr.columns.difference(['y'])]
y=df_df1_corr_re['y']
#train set과 test set분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
lr = linear_model.LinearRegression() #선형회귀 진행
model = lr.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
              
#선형회귀분석 및 OLS 진행
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
model.summary()               
               
               
