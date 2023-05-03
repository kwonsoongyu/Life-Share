#2018년도 전국 오존 농도
import folium
import pandas as pd
from folium import plugins
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

plt.rcParams['font.family']='AppleGothic'
df = pd.read_csv('오존_월별_대기오염도_측정망별_시도별_도시별_측정지점별__20220907002942.csv')

최종 결과물 (분석 코드) 49
# - 있는 행 제거
df=df[~df['2017.06'].str.contains('-')]
df=df[~df['2018.06'].str.contains('-')]
df=df[~df['2019.06'].str.contains('-')]
df=df[~df['2020.06'].str.contains('-')]

#문자 제거
df['2017.06']=df['2017.06'].str.replace('*','')
df['2018.06']=df['2018.06'].str.replace('*','')
df['2019.06']=df['2019.06'].str.replace('*','')
df['2020.06']=df['2020.06'].str.replace('*','')
df['2017.06']=df['2017.06'].astype(float)
df['2018.06']=df['2018.06'].astype(float)
df['2019.06']=df['2019.06'].astype(float)
df['2020.06']=df['2020.06'].astype(float)

#지역별로 그룹화
df_1 = df.groupby(df['구분(2)']).mean().reset_index()

#시군구 json 파일 load
geo = json.load(open('TL_SCCO_CTPRVN.json', encoding='utf-8'))

#맵 생성
map1 = folium.Map(location = [36.1398359, 128.113824],zoom_start = 7)

#시군구 json 파일과 오존 데이터를 매핑해 지도에 시각화
folium.Choropleth(
geo_data=geo,
data=df_1,
columns=['구분(2)', '2018.06'],
key_on='feature.properties.CTP_KOR_NM',
fill_color='YlOrBr',
legend_name='오존농도',
).add_to(map1)
print(map1)
