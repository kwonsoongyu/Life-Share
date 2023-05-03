#C-11
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv('total_results.csv')

#칼럼 추출
df_1 = df.loc[:, ['50대_카테고리1_매출건수(건)','50대_카테고리2_매출건수(건)', '50대_카테고리3_매출건수(건)' ]]

#칼럼 생성
df_1['기준년월'] = ['20.08', '20.09', '20.10', '20.11', '20.12',
'21.01', '21.02', '21.03', '21.04', '21.05', '21.06',
'21.07', '21.08', '21.09', '21.10', '21.11', '21.12',
'22.01','22.02', '22.03', '22.04', '22.05', '22.06',
'22.07', '22.08', '22.09', '22.10', '22.11', '22.12',
'23.01', '23.02', '23.03', '23.04', '23.05', '23.06',
'23.07', '23.08', '23.09', '23.10', '23.11', '23.12']

#Scatter Graph 생성
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=df_1['기준년월'],y=df_1['50대_카테고리1_매출건수(건)'],mode='lines+mark
ers', name='가전렌탈'))
fig.add_trace(go.Scatter(x=df_1['기준년월'],y=df_1['50대_카테고리2_매출건수(건)'],mode='lines+mark
ers', name='카셰어링'))
fig.add_trace(go.Scatter(x=df_1['기준년월'],y=df_1['50대_카테고리3_매출건수(건)'],mode='lines+mark
ers', name='라스트마일모빌리티'))

#그래프 레이아웃 설정 및 출력len(df)
fig.update_layout(title='<b>시간에 따른 50대의 카테고리별 매출건수 비교<b>',
xaxis={'title': {'text': '기준년월'},},
yaxis={'title': {'text': '건수'}},)
fig.show()
