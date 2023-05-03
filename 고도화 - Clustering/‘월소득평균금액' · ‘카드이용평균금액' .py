#필요한 모듈 import
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline
from datetime import datetime
plt.rcParams["font.family"] = 'NanumGothic'
#데이터불러오기 및 컬럼설정
df = pd.read_csv("./import_data/etb_bdt_kcbd_0007.csv", sep="|")
df.columns = ["기준년월","시군구코드", "연령구간코드", "인구수", "남자인구수",
"여자인구수","급여소득자인구수", "자영업자인구수", "기타소득자인구수", "외국인수",
"평균출퇴근거리","하이엔드인구수", "월소득평균금액", "2백만원이하월소득평균금액", "2백초과3백만원이하월소득평균금액", "3백초과4백만원이하월소득평균금액","4백초과5백만원이하월소득평균금액", 
"5백초과6백만원이하월소득평균금액", "6백초과7백만원이하월소득평균금액", "7백초과8백만원이하월소득평균금액",
"8백초과9백만원이하월소득평균금액","9백초과1천만원이하월소득평균금액", "1천만원초과월소득평균금액","중위월소득금액", "주택보유자수",
"총보유주택평균평가금액","승용차보유자수", "수입차보유자수", "승용차신차보유자수", "보유승용차평균가격",
"보유상용차평균가격","카드보유자수", "신용카드보유자수", "신용카드보유평균개수", "체크카드보유자수",
"체크카드보유평균개수 ","카드이용평균금액", "신용판매이용평균금액", "현금서비스평균이용금액", "카드일시불평균이용금액",
"카드할부평균이용금액","신용카드평균이용금액", "체크카드평균이용금액", "해외신용판매평균이용금액",
"해외현금서비스평균이용금액","단기이상연체자수","장기이상연체자수", "대출보유자수", "은행업권대출보유자수", "은행업권외대출보유자수",
"신용대출보유자수","주택담보대출보유자수", "예적금담보대출보유자수", "정책자금대출보유자수", "대출평균잔액",
"은행업권평균대출잔액","은행업권외평균대출잔액", "신용대출평균잔액", "주택담보대출평균잔액", "카드론보유자수",
"카드론평균잔액","한도대출보유자수", "한도대출평균잔액", "당월신규대출자수", "당월신규은행업권대출자수",
"당월신규은행업권외대출자수","당월신규신용대출자수", "당월신규주택담보대출자수", "평균신용평가점수",
"월소득평균2백만원이하대상자수","월소득평균2백초과3백만원이하대상자수","월소득평균3백초과4백만원이하대상자수", 
"월소득평균4백초과5백만원이하대상자수", "월소득평균5백초과6백만원이하대상자수", "월소득평균6백초과7백만원이하대상자수",
"월소득평균7백초과8백만원이하대상자수","월소득평균8백초과9백만원이하대상자수", "월소득평균9백초과1천만원이하대상자수", "월소득평균1천만원초과대상자수"]

#<1차 Clustering>
#1. 표준화
df_labeled=df[['월소득평균금액','카드이용평균금액']]
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df_labeled)
pd.DataFrame(df_scaled)

#2. 클러스터링
model=DBSCAN(eps=0.2,min_samples=10)
clusters=model.fit(df_scaled)

#3. 클러스터링 변수인 cluster 값을 원본 데이터인 df 테이블에 추가
df['cluster']=clusters.labels_

#4. cluster 기준으로 데이터 개수 파악
df.groupby('cluster').count()

#5. clustering 결과 시각화
plt.rcParams["font.family"] = 'NanumGothic'
p=sns.scatterplot(data=df, x="월소득평균금액", y="카드이용평균금액", hue='cluster',palette='Set2')
sns.move_legend(p,"upper right",bbox_to_anchor=(1.15,1.0),title="Clusters",facecolor="white")
sns.set(rc={'axes.facecolor' : '#E5ECF6', 'figure.figsize':(10,8)})
plt.show()

#6. 극단적인 이상치에 속하는 cluster 파악
df[df['cluster'].isin([-1,3])]

#7. 이상치 제거 후 클러스터링 2차 수행
df = df[-df['cluster'].isin([-1,3])] #row 제거하기
df = df.drop('cluster',axis=1) #cluster 열 없애기

#<2차 Clustering>
#8. 표준화데이터
df_labeled=df[['월소득평균금액','카드이용평균금액']]
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df_labeled)
pd.DataFrame(df_scaled)

#9. 클러스터링
model=DBSCAN(eps=0.17,min_samples=10)
clusters=model.fit(df_scaled)

#10. 클러스터링 변수인 cluster 값을 원본 데이터인 df 테이블에 추가
df['cluster_fn']=clusters.labels_

#11. cluster 기준으로 데이터 개수 파악
df.groupby('cluster_fn').count()

#12. clustering 결과 시각화
plt.rcParams["font.family"] = 'NanumGothic'
p=sns.scatterplot(data=df, x="월소득평균금액", y="카드이용평균금액", hue='cluster_fn',palette='Set2')
sns.move_legend(p,"upper right",bbox_to_anchor=(1.15,1.0),title="Clusters",facecolor="white")
sns.set(rc={'axes.facecolor' : '#E5ECF6', 'figure.figsize':(10,8)})
plt.show()

#13. 이상치 제거 후 클러스터링 3차 수행
df = df[-df['cluster_fn'].isin([-1,3])] #row 제거하기
df = df.drop('cluster_fn',axis=1) #cluster 열 없애기

#<3차 Clustering>
#14. 표준화
df_labeled=df[['월소득평균금액','카드이용평균금액']]
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df_labeled)
pd.DataFrame(df_scaled)

#15. 클러스터링
model=DBSCAN(eps=0.18,min_samples=10)
clusters=model.fit(df_scaled)

#16. 클러스터링 변수인 cluster 값을 원본 데이터인 df 테이블에 추가
df['cluster_fn']=clusters.labels_

#17. cluster 기준으로 데이터 개수 파악
df.groupby('cluster_fn').count()

#18. clustering 결과 시각화
plt.rcParams["font.family"] = 'NanumGothic'
p=sns.scatterplot(data=df, x="월소득평균금액", y="카드이용평균금액", hue='cluster_fn',palette='Set2')
sns.move_legend(p,"upper right",bbox_to_anchor=(1.15,1.0),title="Clusters",facecolor="white")
sns.set(rc={'axes.facecolor' : '#E5ECF6', 'figure.figsize':(10,8)})
plt.show()




