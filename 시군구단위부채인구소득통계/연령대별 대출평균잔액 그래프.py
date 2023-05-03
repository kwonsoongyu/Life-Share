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


#기존 대출보유자 대비 신규 대출보유자
## Clustering을 통한 이상치 제거 후 Visualization ##

#df_4(1)<기존 대출보유자수 대비 당월신규대출자수 비교>

#사용할 컬럼 추출
df_1=df.loc[:,['기준년월','대출보유자수','당월신규대출자수']]

#기준년월 필드 date타입으로 변환
df_1['기준년월']=df_1['기준년월'].astype('str')
df_1['기준년월']=df_1['기준년월'].apply(lambda _ :datetime.strptime(_, '%Y%m'))
df_1['기준년월']=df_1['기준년월'].dt.to_period(freq='M')
df_1['기준년월']=df_1['기준년월'].astype('str')

#전처리
df_2=df_1.groupby(by=['기준년월']).sum().reset_index()

#그래프 생성
fig=make_subplots(rows=1, cols=1)
fig.add_trace(go.Bar(name='대출보유자수', x=df_2['기준년월'], y=df_2['대출보유자수'], offsetgroup=0))
fig.add_trace(go.Bar(name='당월신규대출자수', x=df_2['기준년월'], y=df_2['당월신규대출자수'], offsetgroup=0))

fig.update_layout(showlegend=True, title='<b>기존 대출보유자수 대비 당월신규대출자수 비교<b>',xaxis={'title':{'text':'기준년월'}}, barmode='stack')
plotly.offline.plot(fig,filename='df_4(1) 기존 대출보유자수 대비 당월신규대출자수 비교.html')
