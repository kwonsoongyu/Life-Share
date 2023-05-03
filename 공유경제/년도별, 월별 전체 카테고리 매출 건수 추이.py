#C-04
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.rcParams['font.family']='AppleGothic'
df = pd.read_csv('./공유경제(가전렌탈,카셰어링,라스트마일모빌리티).csv')

#histogram Graph 생성 및 출력
fig = px.histogram(df, x="기준년도", y="매출건수(건)", color='기준월',barmode='group',color_discr
ete_sequence=px.colors.qualitative.Set2)
fig.show()
