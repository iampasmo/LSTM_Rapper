#%%
"""
각 아티스트들의 곡수 확인하고,
필요한 아티스트의 가사를 묶어서 한덩어리로 저장하는 코드

"""

#%%

import pandas as pd


#%% 각 아티스트별 곡 수 테이블
    
def artist_count_maker():
    
    # 곡들의 메타데이터 뽑기    
    df_meta_info = pd.read_csv("lyrics_concatnated.csv").iloc[:,:-1]
    
    # 아티스트별 곡 수 뽑기    
    artist_count = df_meta_info.groupby("artist").count()
    artist_count = artist_count["song"].reset_index()
    
    # 정렬
    artist_count = artist_count.sort_values("song", ascending=False).reset_index(drop = True)
    
    return artist_count

artist_count = artist_count_maker()
#%% 제대로 동작했는지 확인
print(artist_count[artist_count["artist"]=="염따"])

#%% 덩어리 파일에서 가사들 불러와서 이어붙이기

def lyrics_extractor_artists(lyrics_table, artist_name = "기리보이"): # lyrics_concatnated.csv 파일 넣어주면 가사만 쭉 이어붙여줌
    
    df = pd.read_csv(lyrics_table) 
    df = df.loc[:,['artist','lyrics']]
    
    
    # 원하는 아티스트의 가사만 뽑아서 쭉 연결
    s = ''
    for row in df.iterrows():    # 데이터 프레임에 대해 각 행마다 접근    
        if row[1]['artist'] == artist_name:            
            s += row[1]['lyrics'] +'//end//' 
    
    # 파일로 저장
    with open("lyrics_by_artists\\texts_all_{}.txt".format(artist_name),
              'w', encoding='utf-8-sig') as textfile1:
        textfile1.write(s)
        
    return s

def load_lyrics_of_artist(artist_name = "기리보이"):
    with open("lyrics_by_artists\\texts_all_{}.txt".format(artist_name),
              "r", encoding='utf-8-sig') as f:
        text = f.read()
        
    return text


#%% 내가 보고싶은 아티스트들만 정해서 가사들 저장

for artist in ["기리보이", "The Quiett", "버벌진트", "염따", "박재범", "에픽하이 (EPIK HIGH)"] :
    lyrics_extractor_artists("lyrics_concatnated.csv", artist)

#%% 제대로 저장 되었는지 확인
s = load_lyrics_of_artist(artist_name = "기리보이")

#%% 저장된 곡수 확인

print(s.count('//end//'))
