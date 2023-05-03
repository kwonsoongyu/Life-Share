#C-02
import numpy as np
import pandas as pd
import plotly.express as px

#csv 읽어오기
df = pd.read_csv('./공유경제(가전렌탈,카셰어링,라스트마일모빌리티).csv')
df3=df.loc[:,['기준월','매출건수(건)']]
df3=df3.groupby(['기준월'],as_index=False)['매출건수(건)'].sum()

#컬럼명 재설정
df3.rename(columns={'기준월':'월'},inplace=True)
x_data = ["1월","10월","11월","12월","2월","3월","4월","5월","6월","7월","8월","9월"]

#파이차트 생성
fig = px.pie(df3,
values='매출건수(건)',
names=x_data,
hole=.4,
color_discrete_sequence=px.colors.sequential.RdBu,
width=500,
height=500)
fig.update_traces(textposition='inside',textinfo='percent+label',
marker=dict(line=dict(color='#000000',width=0)))

#그래프 레이아웃 및 출력
fig.update_layout(
title_text="월별 매출건수",
최종 결과물 (분석 코드) 20
annotations=[dict(text="월별 거래량",x=0.5,y=0.5,font_size=20,showarrow=False)]
)
fig.show()
