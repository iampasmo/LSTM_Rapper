from bs4 import BeautifulSoup
import selenium.webdriver as webdriver
import pandas as pd
import time  
import random

import os

# 구글의 이미지 검색 url 받아옴(아무것도 안 쳤을때의 url)
browser = webdriver.Chrome("C:\\pasmo_python\\chromedriver_win32\\chromedriver.exe")


index= 1  # 차트 검색해서 1번째에 있는애를 보여주는 페이지 가도록 # 50개씩 건너뛰면서 총 34500까지 가야함

switch = True
while switch:
    # 랩/힙합 장르에서 인기순으로 정렬한 페이지에 접근
    browser.get('https://www.melon.com/genre/song_list.htm?gnrCode=GN0300#params%5BgnrCode%5D=GN0300&params%5BdtlGnrCode%5D=&params%5BorderBy%5D=POP&params%5BsteadyYn%5D=N&po=pageObj&startIndex='+str(index))
                
    # 페이지 로드될 때까지 잠시 기다렸다가
    randp=round(random.random()*3,2) # 0~3사이의 랜덤 유리수 추출 (사람처럼 보이게)
    time.sleep(3.5+randp)
    
    # 페이지 html 정보 가져와서 soup으로 파싱
    melon_searched_html = browser.page_source        
    melon_searched_parsed= BeautifulSoup(melon_searched_html, 'html.parser')
    
    table_infos=melon_searched_parsed.select('tbody > tr') # <tbody> </tbody> 바로 아래에 있는 <tr></tr>만 선택
                                                           # > 이 표시는 바로 아래, 공백은 그냥 아래 
                                                           
    data=[]
    for info in table_infos:
        data_element = []
        
        data_element.append(info.find_all("span",{"class": "rank"})[0].get_text()) # 순위
        data_element.append(info.find_all("div",{"class": "ellipsis rank01"})[0].get_text()[1:-1]) # 곡 제목
        data_element.append(info.find_all("div",{"class": "ellipsis rank02"})[0].get_text()[1:-1]) # 아티스트
        data_element.append(info.find_all("div",{"class": "ellipsis rank03"})[0].get_text()[1:-1]) # 앨범명
        data_element.append(info.find_all("span",{"class": "cnt"})[0].get_text()[5:]) # 좋아요 수
        
        
        # 곡 페이지에 접근
        song_id=info.find_all("a",{"class": "btn button_icons type03 song_info"})[0]["href"].split('\'')[1]
        song_url='https://www.melon.com/song/detail.htm?songId=' + song_id
        browser.get(song_url)
        
        #잠깐 기다리고
        randp=round(random.random()*3,2)
        time.sleep(1.5+randp)
        
        # 곡 페이지의 html 파일 받아온다
        song_html = browser.page_source        
        song_parsed= BeautifulSoup(song_html, 'html.parser') # 파싱
        
        # 가사 가져오기
        try:
            lyrics = song_parsed.find_all("div",{"class": "lyric", "id":"d_video_summary"})[0]  
        except:
            lyrics = '가사가 없습니다'
        
        # 곡의 정보 (발매일, 장르)
        song_info=song_parsed.find_all("dl",{"class": "list"})[0]
        date=song_info.find_all("dd")[1].get_text()
        genre = song_info.find_all("dd")[2].get_text()
        
        data_element.append(date)
        data_element.append(genre)
        
        # 가사 아까 가져와둔거 지금 넣기 (제일 마지막열에 넣으려고)
        data_element.append(lyrics)        
        data.append(data_element)
        
        
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(os.path.join("lyrics","lyrics_{:05d}.csv").format(index),
                     header=False, index=False, encoding = 'utf-8-sig')
    print("work : {} / 34500 is completed".format(index))
        
    index += 50 # 한 페이지당 곡 50개씩 있음
    
    if index >500 :
        switch = False

print('finished')

#%%

dataframe.to_csv(os.path.join("lyrics","lyrics_{:05d}.csv").format(index), 
                 header=False, index=False, encoding = 'utf-8-sig')



