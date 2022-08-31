from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import load_model

from PIL import Image ,  ImageOps
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import glob
import tensorflow as tf

# 모델
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 2, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# 이미지 전처리
def imagePrep(path_pattern, WIDTH, HEIGHT, CHANNEL): 
    
    filelist = glob.glob(path_pattern)
    fileabslist = [os.path.abspath(fpath) for fpath in filelist]
    X = []
    for fname in fileabslist:
        img = cv2.imread(fname).astype('float32') / 255
        #print(img.shape)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        if CHANNEL == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        X.append(img)
    X_array = np.array(X)
    return X_array


# 이미지 나타내기
def imageVis(X, W, H ,C):
    fig, ax = plt.subplots(1, 10, figsize=(20, 4))
    for i, img in enumerate(X[:10]):
        ax[i].imshow((img.reshape(W, H, C) *255).astype('uint8'))
    plt.show()
    return fig


#비교 테스트 모델, 학습된 모델 input
def test_model(path ,test_data):
    model = load_model(path)
    ae_imgs = model.predict(test_data)#앞서 만든 오토인코더 모델에 집어 넣습니다.
    random_test = np.random.randint(test_data.shape[0], size=5) 
    plt.figure(figsize=(70, 20))  #출력될 이미지의 크기를 정합니다.
    for i, image_idx in enumerate(random_test):    #랜덤하게 뽑은 이미지를 차례로 나열합니다.
       ax = plt.subplot(2, 7, i + 1) 
       plt.imshow((test_data[image_idx]*255).astype('uint8'))  #테스트할 이미지를 먼저 그대로 보여줍니다.
       ax.axis('off')
       ax = plt.subplot(2, 7, 7 + i +1)
       plt.imshow((ae_imgs[image_idx]*255).astype('uint8'))  #오토인코딩 결과를 다음열에 출력합니다.
       ax.axis('off')
    plt.show()
    return ae_imgs
    
def display_mask(model , i):
    model = np.round(model)
    x = np.array(model[i].reshape(512,512) *255).astype('uint8')
    img = Image.fromarray(x)
    img.show()
    return model

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
    plt.hist(b_img.ravel(), SIZE, [0, SIZE])
    plt.show()
    
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