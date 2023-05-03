#B-02
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#데이터 불러오기
df = pd.read_csv('./공영자전거.csv')
#y축을 2개 사용하기 위한 전처리
fig = make_subplots(specs=[[{"secondary_y": True}]])

#그래프 생성
fig.add_trace(
go.Scatter(x=df['년도'], y=df['자전거보유(대)'], name="자전거보유(대)"),
secondary_y=False,
)

fig.add_trace(
go.Scatter(x=df['년도'], y=df['대여실적(건)'], name="대여실적(건)"),
secondary_y=True,
)

#그래프 레이아웃 설정
fig.update_layout(
title_text="년도별 자전거보유(대), 대여실적(건)"
)

#x축 title
fig.update_xaxes(title_text="기준년월")

#y춛 title
fig.update_yaxes(title_text="자전거보유(대)", secondary_y=False)
fig.update_yaxes(title_text="대여실적(건)", secondary_y=True)
fig.update_xaxes(type='category')
fig.show()
