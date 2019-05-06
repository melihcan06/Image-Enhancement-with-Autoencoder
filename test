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

def ciz(goruntuler):
    for i in range(len(goruntuler)):
        plt.subplot(1,len(goruntuler),i+1)
        plt.imshow(goruntuler[i])
        plt.gray()
        plt.axis('off')
    plt.show()

#yol="uygulamam/yapilar2/"    
yol="yapilar2/"

X_test=np.load(yol+"X_test.npy") 
X_test_noisy=np.load(yol+"X_test_noisy.npy")

X_test = X_test.astype('float32') / 255.
X_test = X_test[..., np.newaxis]

X_test_noisy = X_test_noisy.astype('float32') / 255.
X_test_noisy = X_test_noisy[..., np.newaxis]

model=load_model("model5_3")  
tahmin=model.predict(X_test)
#np.save(yol+"tahmin",tahmin)
boyut=(350,350)
grt=0

for grt in range(10):
  ciz([np.resize(X_test[grt],boyut),np.resize(X_test_noisy[grt],boyut),np.resize(tahmin[grt],boyut)])

  print (psnr(np.resize(X_test[grt],boyut),np.resize(tahmin[grt],boyut)))
