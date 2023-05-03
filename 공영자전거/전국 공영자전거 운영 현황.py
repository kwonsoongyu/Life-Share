#B-01
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams['font.family']='AppleGothic'

#데이터 불러오기
df = pd.read_csv('./공영자전거_운영_현황_20220906200032.csv')
df.columns=['시도별(1)','시도별(2)','운영방식','터미널/주차장설치(개)','자전거보유(대)','대여실적(건)']

#데이터 전처리
df.drop([0,1,2],axis=0,inplace=True)
df.reset_index(inplace=True)
df_1=df.drop(['index'],axis=1)
df_1['터미널/주차장설치(개)']=df_1['터미널/주차장설치(개)'].str.replace('-','0')
df_1['자전거보유(대)']=df_1['자전거보유(대)'].str.replace('-','0')
df_1['대여실적(건)']=df_1['대여실적(건)'].str.replace('-','0')
df_1=df_1.astype({'터미널/주차장설치(개)':int, '자전거보유(대)':int, '대여실적(건)':int})
df_2=df_1.groupby('시도별(1)',as_index=False)['터미널/주차장설치(개)','자전거보유(대)','대여실적(건)'].sum()
labels = df_2['시도별(1)']

#subplots 생성 : Pie subplot을 위한 domain 타입 사용
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels = labels, values=df_2['자전거보유(대)'], name='자전거보유(대)'),1, 1)
fig.add_trace(go.Pie(labels = labels, values=df_2['대여실적(건)'], name='대여실적(건)'),1, 2)

#도넛형태의 Pie 차트를 생성하기 위한 'hole' 사용
fig.update_traces(hole=.3, hoverinfo="label+percent+name",textinfo='percent+label',textposi
tion='inside')
fig.update_layout(
title_text="전국 공영자전거 운영 현황",
annotations=[
dict(text='자전거보유(대)', x=0.165, y=0.5, font_size=15, showarrow=False),
dict(text='대여실적(건)', x=0.83, y=0.5, font_size=15, showarrow=False)])
fig.show()
