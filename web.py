from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image ,  ImageOps
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import glob
import tensorflow as tf
import script as sr

import streamlit as st
import pandas as pd
import numpy as np

# 모델 불러오기
model = load_model('./model/07-0.6855.hdf5')

# 학습된 결과를 출력
def test_model(path, X_test):
    model = load_model(path)
    ae_imgs = model.predict(X_test) 

    # 출력될 이미지의 크기
    fig, ax = plt.subplots((ae_imgs.shape[0]//5+1)*2, 5, figsize=(16,16)) 
    
    # 이미지를 차례대로 나열
    for i, (src, dst) in enumerate(zip(X_test,ae_imgs)): # src=원본, dst=예측 후
        r = i//5 # 행의 인덱스
        c = i%5 # 열 인덱스
        
        # 테스트 이미지 출력
        ax[2*r,c].imshow(src) 
        ax[2*r,c].axis('off')
        
        # 오토인코딩 결과(예측 이미지) 출력
        ax[2*r+1,c].imshow(dst) 
        ax[2*r+1,c].axis('off')
        
    return ae_imgs, fig

# 이미지 불러오기
# 이미지 전처리
def imagePrep(path_pattern, WIDTH, HEIGHT, CHANNEL): 
    filelist = glob.glob(path_pattern)
    fileabslist = [os.path.abspath(fpath) for fpath in filelist]
    X = []
    for fname in fileabslist:
        img = cv2.imread(fname).astype('float32') / 255
        img = cv2.resize(img, (WIDTH, HEIGHT))
        if CHANNEL == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        else:         
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
    X_array = np.array(X)
    
    return X_array


# 오토인코딩 결과를 히스토그램으로 표시 
def displayHist(IMG):
    SIZE = 256
    
    # 이진화
    # 면적계산을 위해 두가지 색상으로 이진화합니다.
    ret, b_img = cv2.threshold(IMG, 128, 255, cv2.THRESH_BINARY)
    
    # 히스토그램 계산
    hist = cv2.calcHist(images = [b_img], 
                       channels = [0], 
                       mask = None,
                       histSize = [SIZE],
                       ranges = [0, SIZE])
    
    # 출력
#     plt.hist(b_img.ravel(), SIZE, [0, SIZE])
#     plt.show()
    
    return b_img, hist

# 면적 계산
def calArea(IMG):
    b_img, hist = displayHist(IMG)
    height, width = b_img.shape[0], b_img.shape[1] 
    rectangle_area = height * width
    rate_w = hist[-1] / rectangle_area
    rate_b = hist[0] / rectangle_area

    ds_area = 100*100 # 전체 면적
    ds_S = ds_area*rate_w # 질병 면적
    st_S = ds_area*rate_b # 정상 면적
    
    return ds_S 


def display_mask(model ,i):
#     """Quick utility to display a model's prediction."""
    mask = np.argmax(model[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = tf.keras.preprocessing.image.array_to_img(mask)
    
    #PIL -> Cv2 사용을 위해서 변환
    np_img = np.array(img)
    
    #medianBlur 이미지 잡음 처리
    dst = cv2.medianBlur(np_img, 3)
    ret, b_img = cv2.threshold(dst, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('b_img', b_img)
    
    #이미지 저장
    #cv2.imwrite(f'./100img/{name}.jpg',dst)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    return b_img

        
# 업로드 된 이미지 저장하는 함수    
# 디렉토리와 파일 주면, 해당 디렉토리에 파일 저장
def save_uploaded_file(directory, file) :
    if not os.path.exists(directory): #  디렉토리 없으면 만들기
        os.makedirs(directory)
    with open(os.path.join(directory, file.name), 'wb') as f: # 파일 저장하기
        f.write(file.getbuffer())
    return st.success('Saved file : {} in {}' .format(file.name, directory))



## 페이지 이름 지정
st.set_page_config(
     page_title="🌲 Pine wilt disease Web 🌲",
     layout="wide",
     initial_sidebar_state="expanded"
 )

## 로고 이미지 넣기
# st.image("./title.png", use_column_width=True)


## 사이드바
# data = pd.read_csv("pine.csv") # 전국 고사목 수
# data = data.set_index("년도") # 년도 데이터를 인덱스로 지정
# chart_data = pd.DataFrame(data)

st.sidebar.header("소나무재선충병")
name = st.sidebar.selectbox("Menu", ['개요', '면적 추출 예시', '면적 계산하기'])

# 1) 사이드바 1번 - 전국 고사목 수
if name == "개요":
    st.write("### 🌳 소나무재선충병 피해 본수와 투입 예산")
    # st.image("./그림2.jpg", width=500)
    st.write("""
    ### 🌳 소나무재선충병이란?
    - 매개충인 하늘소에 의해 빠른 확산이 이루어짐
    - 상처 부위를 통해 침입한 재선충이 소나무의 수분·양분의 이동통로를 막아 나무를 죽게 하는 병
    - 치료약이 없어 감염되면 100% 고사
    - 지금까지 재선충에 의해 고사한 소나무류는 총 1,200만 본
    - 22년 4월 말 기준 피해목이 작년과 비교해 22.6% 증가
    """)

    st.write("### 🌳 전국 고사목 수(~2020년)")
    st.bar_chart(chart_data)
    # 2022년 현황
    st.write("### 🌳 2022년도 현황")
    col1, col2 = st.columns(2)
    col1.metric(label="발생 시·군·구", value="135개", delta="4개 지역")
    col2.metric(label="고사목 수", value="38만 본", delta="22.6%")

# 2) 사이드바 2번 - 면적 추출 예시    
if name =="면적 추출 예시":
    st.markdown("#### 🌳 소나무재선충병 이미지")
    st.write("- 산림 모형과 전처리 이미지")
 
    # col1, col2, col3, col4, col5 = st.columns(5)
    # col1.image("./train/001.jpg", width=200)
    # col2.image("./train/002.jpg", width=200)
    # col3.image("./train/003.jpg", width=200)
    # col4.image("./train/004.jpg", width=200)
    # col5.image("./train/005.jpg", width=200)    
  
    # col1.image("./target/target001.jpg", width=200)
    # col2.image("./target/target002.jpg", width=200)
    # col3.image("./target/target003.jpg", width=200)
    # col4.image("./target/target004.jpg", width=200)
    # col5.image("./target/target005.jpg", width=200)

# 3) 사이드바 3번 - 면적 계산하기    
# 사진 업로드(여러장도 가능하게)        
if name =="면적 계산하기":
    st.subheader("🌳 면적 계산하기")

    uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True,
                                  type=["png","jpg","jpeg"])
    for i in uploaded_files :
    # 업로드 된 사진 저장하기
        save_uploaded_file('temp_files', i)
    
    
    # 저장된 사진에 대하여..
    # 1) 이미지 전처리
    HEIGHT, WIDTH, CHANNEL = 512, 512, 3
    # X_test = imagePrep('./temp_files/*.jpg', WIDTH, HEIGHT, CHANNEL)  
    
    # 2) 원본과 전처리 후 보여주기 + 면적 출력
    model_path = './model/07-0.6855.hdf5'
    y_pred, fig = test_model(model_path, X_test)
    st.pyplot(fig)
    
    img = display_mask(y_pred, 0)
    calArea(img)
    
    



    

    
