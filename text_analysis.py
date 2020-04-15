#%%
"""
아티스트의 총 가사 길이, 토큰 개수 확인하는 코드
"""

#%% 모듈 불러오기
import numpy as np
import re

from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

#%%

class Text_Analyzer:
        
    def __init__(self):
        pass
    
    def text_load(self, text_file = 'texts_all.txt' ):
                
        with open(text_file, encoding='utf-8-sig') as f:
            text = f.read()            
        
        text = re.sub('\n'   , '', text) # 줄바꿈 다 없앰
        text = re.sub(' {2,}',' ', text) # 공백 두개 이상이면 한 개로 바꿈
        
        # 특수문자 처리는 우선은 하지 않고 냅둔다
        #text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text) # 두번째 입력변수에 r' \1 ' 대신 ' \\1 ' 써줘도 똑같음 - r 써주는 방법이 더 안전할 수 있대        
        
        self.text = text
        
    def text_load_artist(self, artist = "기리보이"):        
        self.text_load(text_file = "lyrics_by_artists\\texts_all_{}.txt".format(artist))
    
    
    def tokenize(self):
        # 토큰화
        self.tokenizer = Tokenizer(char_level = False, filters = '') # 필터에 적혀있는 애는 토큰으로 안만든대
        self.tokenizer.fit_on_texts([self.text])                   # 여기서 리스트 하나 더 안씌우면 문자단위로 토큰화 한다.
        self.total_words = len(self.tokenizer.word_index) + 1           # 모든 단어종류 +1 // 1 더하는 이유는, 토크나이저.word_index가 1부터 인덱싱 해서, 뒤에 to_categorical로 원핫 인코딩할때 맞춰주기 위해서임
        self.token_list = self.tokenizer.texts_to_sequences([self.text])[0]  # text의 모든 단어들을 토큰으로 바꾼거
        
        self.training_size = len(self.token_list) 
        
    def counter(self):
        words_counts = self.tokenizer.word_counts.items()
        words_counts = sorted(words_counts, key = lambda x :x[1], reverse = True)
                
        # [ten_words, five_words, one_words]
        for i, word in enumerate(words_counts):
            if words_counts[i][1]<=10 and words_counts[i-1][1]>10:
                ten_words = i
                
            if words_counts[i][1]<=5 and words_counts[i-1][1]>5:
                five_words = i
            
            if word[1] == 1:
                one_words = i
                break        
        
        # n회 이상 등장한 단어들이 전체 단어에서 차지하는 비율 반환
        a = 1 - np.array([ten_words,five_words, one_words]) / len(words_counts)
        # 총 단어 개수 반환
        b = len(words_counts)
        
        # 가사 길이도 함께 반환
        return [a,b, self.training_size]
        
#%%
# 뭉텅이 함수
def analyzer_artists_1(Text_Analyzer, artist):
    t1 = Text_Analyzer()    
    t1.text_load_artist(artist)
    t1.tokenize()    
    return t1.counter()

# 각 아티스트별 단어분포, 단어집합의 개수 출력
for artist in ["기리보이", "The Quiett", "버벌진트", "염따", "박재범", "에픽하이 (EPIK HIGH)"]:
    np.set_printoptions(precision=2)
    print(artist, ":", "{}".format(analyzer_artists_1(Text_Analyzer, artist) ))
np.set_printoptions(precision=None)


















