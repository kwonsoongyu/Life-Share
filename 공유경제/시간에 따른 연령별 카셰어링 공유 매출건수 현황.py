#C-06
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
최종 결과물 (분석 코드) 24
from plotly.subplots import make_subplots
df = pd.read_csv('total_results.csv')

#칼럼 추출
df_1 = df.loc[:, ['20대_카테고리2_매출건수(건)',
'30대_카테고리2_매출건수(건)',
'40대_카테고리2_매출건수(건)',
'50대_카테고리2_매출건수(건)']]

#새로운 칼럼추출
df_1['기준년월'] = ['20.08', '20.09', '20.10', '20.11', '20.12',
'21.01', '21.02', '21.03', '21.04', '21.05', '21.06',
'21.07', '21.08', '21.09', '21.10', '21.11', '21.12',
'22.01','22.02', '22.03', '22.04', '22.05', '22.06',
'22.07', '22.08', '22.09', '22.10', '22.11', '22.12',
'23.01', '23.02', '23.03', '23.04', '23.05', '23.06',
'23.07', '23.08', '23.09', '23.10', '23.11', '23.12']

#바그래프 생성
fig = go.Figure()
fig.add_trace(go.Bar(
x=df_1['기준년월'],
y=df_1['20대_카테고리2_매출건수(건)'],
name='20대 카셰어링',
marker=dict(
color='rgb(252, 180, 174)',
line=dict(color='rgb(242, 209, 209)',)
)
))
fig.add_trace(go.Bar(
x=df_1['기준년월'],
y=df_1['30대_카테고리2_매출건수(건)'],
name='30대 카셰어링',
marker=dict(
color='rgb(202, 184, 255)',
line=dict(color='rgb(201, 184, 255)',)
)
))
fig.add_trace(go.Bar(
x=df_1['기준년월'],
y=df_1['40대_카테고리2_매출건수(건)'],
name='40대 카셰어링',
marker=dict(
color='rgb(39, 123, 192)',
line=dict(color='rgb(202, 184, 255)',)
)
))
fig.add_trace(go.Bar(
x=df_1['기준년월'],
y=df_1['50대_카테고리2_매출건수(건)'],
name='50대 카셰어링',
marker=dict(
color='rgb(255, 178, 0)',
line=dict(color='rgb(202, 184, 255)',)
)
))

#그래프 레이아웃 설정 및 출력
fig.update_layout(title='<b>시간에 따른 연령별 카셰어링 공유 매출건수 현황<b>', barmode='stack')
fig.show()
