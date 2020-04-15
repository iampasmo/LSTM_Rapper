#%%
import os
import glob

import pandas as pd

#%% 각 파일들의 데이터 쭉 합치기 (전체 파일)

# 불러올 파일들 목록 뽑아놓고
file_names = glob.glob("lyrics\\lyrics*.csv")
file_names.sort()

# 시작점
df = pd.read_csv(os.path.join('lyrics','lyrics_00001.csv'), 
                names = ['number','song','artist','album','likes','dates','genre','lyrics'], 
                header = None)
# 뒤에 이어 붙이기
for small_file in file_names[1:]:
    small_df = pd.read_csv(small_file, 
                           names = ['number','song','artist','album','likes','dates','genre','lyrics'], 
                           header = None)
    df = pd.concat([df,small_df])
    print('{} is finishied'.format(small_file))
    

#%% 데이터 편집

# 아티스트 이름 두번 나오는거 수정
df['artist'] = df['artist'].apply( lambda x: x[:len(x)//2])

#%% 데이터 편집 2
# 가사 시작시 불필요한 단어 등장하는거 제거
# 가사에 "<br/>"를 " <br/> " 로 한칸씩 띄움 -> 토큰화시킬수 있도록 
# 제일 마지막에 </div> 제거
def lyrics_cleaner(lyrics):
    if lyrics =='가사가 없습니다':
        return ' '
    else :
        lyrics = lyrics.replace("\t\t\t\t\t\t\t", "")
        lyrics = lyrics.replace('<div class="lyric" id="d_video_summary"><!-- height:auto; 로 변경시, 확장됨 -->', "")
        lyrics = lyrics.replace("<br/>", " <br/> ")
        lyrics = lyrics.replace("</div>", "")
        
        return lyrics
    

df['lyrics'] = df['lyrics'].apply( lyrics_cleaner )



#%% 합친 데이터 저장


df.to_csv("lyrics_concatnated.csv",
          header=True, index=True, encoding = 'utf-8-sig')


#%%
print(df.head())
