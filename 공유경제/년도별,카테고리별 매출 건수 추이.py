#C-01
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#csv 읽어오기
df = pd.read_csv('./공유경제(가전렌탈,카셰어링,라스트마일모빌리티).csv')

#칼럼 추출
df_1=df.loc[:,['기준년도','카테고리','매출건수(건)']] #기준월보다 보기 깔끔함
df_1=df_1.groupby(['기준년도','카테고리'],as_index=False).mean()

#line Graph 생성
fig=px.line(df_1,
x="기준년도",
y="매출건수(건)",
color="카테고리",
line_group="카테고리",
hover_name="카테고리",
title="월별 카테고리 매출건수 추이",
width=1000,
height=600)

fig.update_layout(plot_bgcolor="#F7F7F7")
fig.update_xaxes(type='category')
최종 결과물 (분석 코드) 19
fig.show()
