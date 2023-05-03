#C-12
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('total_results.csv')
df_1 = df.loc[:, ['남_카테고리1_매출금액(천원)','남_카테고리2_매출금액(천원)',
'남_카테고리3_매출금액(천원)' ]]
df_1['기준년월'] = ['20.08', '20.09', '20.10', '20.11', '20.12',
'21.01', '21.02', '21.03', '21.04', '21.05', '21.06',
'21.07', '21.08', '21.09', '21.10', '21.11', '21.12',
'22.01','22.02', '22.03', '22.04', '22.05', '22.06',
'22.07', '22.08', '22.09', '22.10', '22.11', '22.12',
'23.01', '23.02', '23.03', '23.04', '23.05', '23.06',
'23.07', '23.08', '23.09', '23.10', '23.11', '23.12']
df_2 = df_1.sum().reset_index()
df_2.drop(index = df_2.index[-1], axis = 0, inplace = True)
df_2['카테고리'] = ['가전렌탈', '카셰어링', '라스트마일모빌리티']

#그래프 레이아웃 및 출력
fig = px.bar(df_2, x="카테고리", y=df_2[0], color="카테고리", title='<b>남성 카테고리별 매출금액</b
>')
fig.show()
