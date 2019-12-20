import sys
sys.path.insert(0,"uygulamam/")
from veriseti_hazirlama import *
import keras
import cv2
import numpy as np
from keras.layers import Dropout, Activation, Flatten
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Sequential,Model
from keras.models import load_model


#train datasini grayscale e donusturuyor
def goruntuleri_don(veri):  
  goruntuler=[]
  
  for i in veri:  
    goruntuler.append(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY))
    
  return goruntuler

#yol="uygulamam/yapilar2/"    
yol="yapilar2/"


#TRAIN VERILERI
X_train=np.load(yol+"X_train.npy") 
X_train_noisy=np.load(yol+"X_train_noisy.npy")

X_train = X_train.astype('float32') / 255.
X_train = X_train[..., np.newaxis]

X_train_noisy = X_train_noisy.astype('float32') / 255.
X_train_noisy = X_train_noisy[..., np.newaxis]


#TEST VERILERI
X_test=np.load(yol+"X_test.npy") 
X_test_noisy=np.load(yol+"X_test_noisy.npy")

X_test = X_test.astype('float32') / 255.
X_test = X_test[..., np.newaxis]

X_test_noisy = X_test_noisy.astype('float32') / 255.
X_test_noisy = X_test_noisy[..., np.newaxis]


input_img = Input(shape=(350, 350, 1)) 

#encoder
conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_img) 
conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2) 
conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)

# decoder
conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3) 
up1 = UpSampling2D((2,2))(conv4) 
conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(up1)
conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(conv5)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(conv6)
#decoded = ZeroPadding2D(padding=(1,1))(conv7)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')# loss='binary_crossentropy'
autoencoder.fit(X_train_noisy, X_train,epochs=50,batch_size=8,shuffle=True,validation_split=0.2)#validation_data=(X_test_noisy, X_test)

autoencoder.save("model5_3")
