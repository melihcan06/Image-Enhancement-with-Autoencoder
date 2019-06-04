import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"uygulamam/")
from veriseti_hazirlama import *
from sklearn.model_selection import train_test_split

def goruntuleri_don(veri):
    goruntuler = []

    for i in veri:
        goruntuler.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))

    return goruntuler
  
def verileri_arraye_cevir():
  X_train=[]
  for i in range(1,523):#1,523
      img=cv2.imread("uygulamam/yapilar2/"+str(i)+".jpg",0)
      X_train.append(img)
  X_train=np.array(X_train)
  
  return X_train

#verinin cekilmesi

X_train_ham=verileri_arraye_cevir()
X_train_ham,X_test=train_test_split(X_train_ham,test_size=0.2)

X_train = []
#arka arkaya 3 tane verilmesi
for i in range(3):
    for j in range(X_train_ham.shape[0]):
        X_train.append(X_train_ham[j])
X_train=np.array(X_train)

#gurultu uygulanmasi
X_train_noisy,yntmler=yontemleri_uygula_coklu2(X_train)
X_train_noisy_ham=np.array(X_train_noisy)

X_test_noisy,yntmler=yontemleri_uygula_coklu2(X_test)
X_test_noisy_ham=np.array(X_test_noisy)


np.save("uygulamam/yapilar2/X_train",X_train)
np.save("uygulamam/yapilar2/X_test",X_test)
np.save("uygulamam/yapilar2/X_train_noisy",X_train_noisy)
np.save("uygulamam/yapilar2/X_test_noisy",X_test_noisy)
