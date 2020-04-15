#%% 모듈 불러오기
import numpy as np
import re

from keras.layers import Dense, LSTM, Input, Embedding, Dropout
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import LambdaCallback

import os
import pickle


#%%


#%%

class text_LSTM:
    def __init__(self, artist_name= "염따"):
        self.seq_length = 20      
        
        self.artist_name = artist_name
    
    
    def text_loader(self) :            
        
        text_file = os.path.join("lyrics_by_artists","texts_all_{}.txt".format(self.artist_name))
        
        with open(text_file, encoding='utf-8-sig') as f:
            text = f.read()
        
        self.start_story = '| '  * self.seq_length
        
        # 텍스트 정제    
        text = text.lower()
        text = self.start_story + text
        text = text.replace('//end//', self.start_story) # 끝나면 새로 시작하는 거 표시
                
        text = re.sub('\n'   , '', text) # 줄바꿈 다 없앰 - 진짜 줄바꿈은 <br/>로 남아있음
        text = re.sub(' {2,}',' ', text) # 공백 두개 이상이면 한 개로 바꿈
        
        # 특수문자 처리는 우선은 하지 않고 냅둔다
        #text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text) # 두번째 입력변수에 r' \1 ' 대신 ' \\1 ' 써줘도 똑같음 - r 써주는 방법이 더 안전할 수 있대        
        
        self.text = text       
        
    ###
    def tokenize(self):
        # 토큰화
        self.tokenizer = Tokenizer(char_level = False, filters = '') # 필터에 적혀있는 애는 토큰으로 안만든대
        self.tokenizer.fit_on_texts([self.text])                   # 여기서 리스트 하나 더 안씌우면 문자단위로 토큰화 한다.
        self.total_words = len(self.tokenizer.word_index) + 1           # 모든 단어종류 +1 // 1 더하는 이유는, 토크나이저.word_index가 1부터 인덱싱 해서, 뒤에 to_categorical로 원핫 인코딩할때 맞춰주기 위해서임
        self.token_list = self.tokenizer.texts_to_sequences([self.text])[0]  # text의 모든 단어들을 토큰으로 바꾼거
        
        self.training_size = len(self.token_list) - self.seq_length
        
    
    ### seq_length만큼 잘라서, 신경망에 학습시킬 수 있도록 각 seq - 다음단어 쌍을 만들어준다.    
    def generate_sequences(self, step = 1):
        X = []
        y = []
                
        #for i in dice:
        for i in range(self.training_size):
            X.append(self.token_list[i:i+self.seq_length])
            y.append(self.token_list[i + self.seq_length])
            
        y = np_utils.to_categorical(y, num_classes = self.total_words) 
        # 1차원짜리 y값을 원핫 인코딩으로 변환 (total_words 개수만큼)
        # to_categorical 얘는 그냥 0이면 1번쨰열을 1 나머지는 0, 55면 56번째 열을 1 나머지는 0 이런식으로
        # 숫자 값 그대로 원핫 인코딩 해주는 애        
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        print('시퀀스 개수:', self.training_size, '\n')
        
        #return [X, y]        
        
    ###
    
    def build_network(self):
        
        n_units = 256
        embedding_size = 100 # 100이랑 50이랑 각각 테스트 해보자
        
        # 여기서부터 신경망 시작
        
        text_in = Input(shape = (None,))
        x = Embedding(self.total_words, embedding_size)(text_in)
        x = LSTM(units = n_units, return_sequences = True)(x)   # lstm 한층 더 쌓기         
        x = Dropout(rate = 0.1)(x)
        x = LSTM(units = n_units)(x)    # output의 차원은 n_units 개        
        text_out = Dense(units = self.total_words, activation = 'softmax')(x)   # 아웃풋은 단어 종류의 개수만큼 표현이 되어야 하니까
        
        self.model1 = Model(text_in, text_out)
        
        
    def compile_network(self):
        
        optimizer = RMSprop(lr = 0.001)
        self.model1.compile(optimizer = optimizer, 
                            loss = 'categorical_crossentropy', 
                            metrics = ['accuracy'])
        
        
    def fit_network(self, epochs = 10, batch_size = 32):
        
        history = self.model1.fit(x = self.X, 
                                y = self.y, 
                                epochs = epochs, 
                                batch_size = batch_size, 
                                shuffle = True)        
        return history
    
    
    ### 모델 불러오기        
    def load_network(self, model_file_path = 'saved_models\\text_model.h5'):
        print('load model...')
        self.model1 = load_model(model_file_path)
        print('model loaded : {0}'.format(model_file_path))        

    
    # 학습된 신경망을 통해 가사 생성  
    def generate_text(self, seed_text, next_words, max_sequence_len=20 , temperature = 1):
        
        output_text = seed_text + ' '
        seed_text = self.start_story + seed_text + ' '
        
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]  # 글자들을 인덱스로 바꾼다. seed_text 겉에 리스트 씌워야 한다. 역으로 sequences_to_texts 할 때도 겉에 리스트 씌워야 한다.
            token_list = token_list[-max_sequence_len:]
            token_list = np.reshape(token_list, (1, max_sequence_len))  # model1.predict 에 넣어주려면 배치 형태로 넣어줘야 하니까 앞에 차원 하나 만들어줌
            #print(type(token_list)) # numpy array
            
            probs = self.model1.predict(token_list, verbose = 0)[0]  # 배치형태로 값이 들어갔으니까, 출력도 배치형태라서 [0] 으로 차원 하나 풀어준다.
            y_index = self._sample_with_temp(probs, temperature = temperature)
            
            output_word = self.tokenizer.index_word[y_index] if y_index > 0 else ''
            # 신경망을 통해 뽑은 숫자를 글자로 다시 바꿔준다
            # if절에서 y_index가 0보다 커야하는 조건 넣어준 이유는, tokenizer.index_word 딕셔너리는 key값 0이 없기 때문.
            
            if output_word == '|':
                break
            
            output_text += output_word + ' '
            seed_text += output_word + ' '            
            
        return output_text
    
    
    def _sample_with_temp(self, preds, temperature = 1.0):
        # 확률 배열에서 인덱스 하나를 샘플링하는 함수
        # preds에는 원핫 카테고리별 확률값이 들어있는, 확률분포 리스트가 들어와야해.
        
        preds = np.array(preds).astype('float64')
        exp_preds = np.exp( np.log(preds) / temperature ) 
        preds = exp_preds / np.sum(exp_preds)  # 이게 확률분포
        probability = np.random.multinomial(1,preds,1) # 입력변수 각각 (주사위 몇번 던질래, 주사위 눈은몇개고 확률은 각각 어떻게 돼?, 이 결과값이 몇개 필요해?)
        
        return np.argmax(probability)
    
    
## 단어 등장 확률 보고 싶어서 만듬
    
    def _show_preds(self,preds):     
            
        preds = enumerate(preds)
        preds = sorted(preds, key = lambda x:x[1], reverse = True )[:10]
        
        for i in range(10):
            print('{0} : {1:10.2%} '.format(self.tokenizer.index_word[ preds[i][0] ], preds[i][1]))
            
    
    def generate_text2(self, seed_text, next_words=1, max_sequence_len=20 , temperature = 1):
        
        output_text = seed_text + ' '
        seed_text = self.start_story + seed_text
        
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]  # 글자들을 인덱스로 바꾼다. seed_text 겉에 리스트 씌워야 한다. 역으로 sequences_to_texts 할 때도 겉에 리스트 씌워야 한다.
            token_list = token_list[-max_sequence_len:]
            token_list = np.reshape(token_list, (1, max_sequence_len))
            #print(type(token_list)) # numpy array
            
            probs = self.model1.predict(token_list, verbose = 0)[0]
            self._show_preds(probs)

############ 이 위에까지는 구조, 이 밑에 부터는 실질적으로 실행하는 함수들 #####

    #클래스 불러오고, 가사 파일 토큰화
    def func_make_structure(self):
        
        self.text_loader()
        self.tokenize()
        self.generate_sequences()    
        
        # 신경망 만들기
        self.build_network()
        self.compile_network()
        
        #% 토큰화된 단어들 보기
        print("tokens : ")
        tmp = 0
        for i,j in self.tokenizer.index_word.items():
            
            print(i,':',j)
            tmp += 1
            if tmp > 30 : break
        
        print("total words : ", self.total_words)
        print("training_size : ",  self.training_size )
        

    # 훈련시키고 모델 저장, 훈련 기록도 함꼐 저장    
    def func_train(self, epochs_train = 150):        
        
        history_y = []
        for i in range(1):
            history_y.append( self.fit_network(epochs = epochs_train, batch_size = 32) ) 
            self.model1.save("trained_data\\{0}_{1}epoch.h5".format(self.artist_name, epochs_train*(i+1)))
            
        # 훈련기록 피클 파일로 저장
        with open ("trained_data\\history_{}.pkl".format(self.artist_name),"wb") as file1:
            pickle.dump(history_y, file1)

    


    
    



    # 훈련된 모델 불러오기
    def func_load(self, epochs_load = 300):                
        self.load_network("trained_data\\{0}_{1}epoch.h5".format(self.artist_name, epochs_load))
        
    # 랩 생성
    def func_rap(self, seed_text = "난 배가 고파", temperature = 3):            
        print(self.generate_text(seed_text = seed_text ,
                               next_words = 100,
                               max_sequence_len=20 , 
                               temperature = temperature
                               ) .replace('<br/>','\n') )
# 이어질 단어들 확률 보여주기
        
    def func_show_prob(self, seed_text = "난 배가 고파"):
        self.generate_text2(seed_text, 1)






