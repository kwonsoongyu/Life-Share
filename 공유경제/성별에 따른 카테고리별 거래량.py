#C-03
import numpy as np
import pandas as pd
import plotly.express as px

#csv 읽어오기
df = pd.read_csv('./공유경제(가전렌탈,카셰어링,라스트마일모빌리티).csv')
df3=df.loc[:,['성별','매출건수(건)','카테고리']]
df3=df3.groupby(['카테고리','성별'],as_index=False)['매출건수(건)'].sum()

#카테고리 및 성별로 그룹화
df4=df3.groupby(by=['카테고리','성별']).sum().reset_index()
df5=df4[df4['성별']=='여']
df6=df4[df4['성별']=='남']

#바 그래프 생성
trace1=go.Bar(x=df5['카테고리'], y=df5['매출건수(건)'], name="여자")
trace2=go.Bar(x=df6['카테고리'], y=df6['매출건수(건)'], name="남자")

#그래프 레이아웃 및 출력
data=[trace1, trace2]
layout=go.Layout(title='카테고리별 성비 비교')
최종 결과물 (분석 코드) 21
fig=go.Figure(data=data, layout=layout)
fig.show()
