## Clustering을 통한 이상치 제거 후 Visualization ##

#<연령대별 대출평균잔액 그래프>

#전처리
df_1=df.loc[:,['연령구간코드','대출평균잔액']]
df_1['연령구간코드']=df_1['연령구간코드']//10
df_1['연령구간코드']=df_1['연령구간코드']*10
df_2=df_1.groupby(by=['연령구간코드']).mean().reset_index()

#그래프생성
colors = sns.color_palette('hls', len(df_2['연령구간코드'])).as_hex()
trace2=go.Bar(x=df_2['연령구간코드'], y=df_2['대출평균잔액'], marker={'color':colors}, width=5)
data=[trace2]
fig=go.Figure(data=data, layout=layout)
layout=go.Layout(title='<b>연령대별 대출평균잔액<b>')
plotly.offline.plot(fig,filename='df_2(1) 연령대별 대출평균잔액 막대그래프.html')
