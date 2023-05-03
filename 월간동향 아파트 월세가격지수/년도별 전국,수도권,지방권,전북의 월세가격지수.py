#D-01
import pandas as pd
from pandas import Series, DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv('./아파트월세가격지수.csv')

#데이터 프레임 생성
result = DataFrame()

#데이터 칼럼 추출
df1 = df[['Jun-15','Jul-15','Aug-15',"Sep-15",'Oct-15','Nov-15','Dec-15']]
df1 = df1.mean(axis=1)
df1 = pd.DataFrame(df1)
최종 결과물 (분석 코드) 37
result['15년'] = df1

df2 = df[['Jan-16','Feb-16','Mar-16','Apr-16','May-16','Jun-16','Jul-16','Aug-16',"Sep-16",'Oct-16','Nov-16','Dec-16']]
df2 = df2.mean(axis='columns')
df2 = pd.DataFrame(df2)
result['16년'] = df2
          
df3 = df[['Jan-17','Feb-17','Mar-17','Apr-17','May-17','Jun-17','Jul-17','Aug-17',"Sep-17",'Oct-17','Nov-17','Dec-17']]
df3 = df3.mean(axis=1)
df3 = pd.DataFrame(df3)
result['17년'] = df3
          
df4 = df[['Jan-18','Feb-18','Mar-18','Apr-18','May-18','Jun-18','Jul-18','Aug-18',"Sep-18",'Oct-18','Nov-18','Dec-18']]
df4 = df4.mean(axis=1)
df4 = pd.DataFrame(df4)
result['18년'] = df4
          
df5 = df[['Jan-19','Feb-19','Mar-19','Apr-19','May-19','Jun-19','Jul-19','Aug-19',"Sep-19",'Oct-19','Nov-19','Dec-19']]
df5 = df5.mean(axis=1)
df5 = pd.DataFrame(df5)
result['19년'] = df5
          
df6 = df[['Jan-20','Feb-20','Mar-20','Apr-20','May-20','Jun-20','Jul-20','Aug-20',"Sep-20",'Oct-20','Nov-20','Dec-20']]
df6 = df6.mean(axis=1)
df6 = pd.DataFrame(df6)
result['20년'] = df6
          
df7 = df[['Jan-21','Feb-21','Mar-21','Apr-21','May-21','Jun-21']]
df7 = df7.mean(axis=1)
df7 = pd.DataFrame(df7)
result['21년'] = df7
          
df8 = df[['Aug-16','Aug-17','Aug-18',"Aug-19",'Aug-20']]
df8 = df8.mean(axis=1)
df8 = pd.DataFrame(df8)
result['22년'] = df8
          
#잔차 평균값 최소 최대 등 확인
world = result.loc[[0,1,2,181]]
          
#그래프 만들기
line_color = '#79db93'
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=result.columns,y=result.loc[0],mode='lines+markers', name='전국'))
fig.add_trace(go.Scatter(x=result.columns,y=result.loc[1],mode='lines+markers', name='수도권'))
fig.add_trace(go.Scatter(x=result.columns,y=result.loc[2],mode='lines+markers', name='지방권'))
fig.add_trace(go.Scatter(x=result.columns,y=result.loc[36],mode='lines+markers', name='전북'))
fig.update_layout(title='<b>월세 동향<b>',
xaxis={'title': {'text': '년도'},},
yaxis={'title': {'text': '가격지수'}},)
world.describe()
