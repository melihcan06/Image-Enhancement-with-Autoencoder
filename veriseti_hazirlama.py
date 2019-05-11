import numpy as np
import cv2
from yontemler import *

def random_float(bas,son,adet):
    a = [np.random.random_integers(bas, son, adet)[i] + round(np.random.random(adet)[i],2) for i in range(adet)]
    if a == 0:
        a = 0.01
    return a

def kernel_don():
    a = [3,3,3,3,3,5,5,5,5,7,7,7,9,9,11,15]
    b = a[np.random.random_integers(0,len(a)-1,1)[0]]
    return (b,b)

def yeginlik_deger_don(x1_aralik=84,y1_aralik=84,x2_aralik=170,y2_aralik=170,baslangic_aralik=60):
    baslangic = np.random.random_integers(0,baslangic_aralik,1)[0]
    x1 = np.random.random_integers(0,x1_aralik,1)[0]
    y1 = np.random.random_integers(0, y1_aralik, 1)[0]
    x2 = np.random.random_integers(x1_aralik+1, x2_aralik, 1)[0]
    y2 = np.random.random_integers(y1_aralik+1, y2_aralik, 1)[0]

    return x1,y1,x2,y2,baslangic

def yon1(grt):
    return gaussian_noise(grt)

def yon2(grt):
    return salt_pepper(grt)

def yon3(grt):
    gamma_random = random_float(0.4, 2.5, 1)[0]
    return gamma(grt, gamma_random)

def yon4(grt):
    log_random = random_float(10, 100, 1)[0]
    return log_transform(grt, log_random)

def yon5(grt):
    kernel_size = kernel_don()
    blur_degeri = np.random.random_integers(0, 2, 1)[0]
    return blur(grt, blur_degeri, kernel_size)

def yon6(grt):
    x1, y1, x2, y2, baslangic = yeginlik_deger_don()
    return yeginlik_seviyesi_dilimleme(grt, x1, y1, x2, y2, baslangic)

def yontem_don(uygulanacak_yontem_sayisi):
    yontemler=[]
    while True:
        a = np.random.random_integers(1,6,1)[0]
        if len(yontemler) == uygulanacak_yontem_sayisi:
            return yontemler
        if yontemler.count(a) == 0:
            yontemler.append(a)

def yontemleri_uygula(grt):
    uygulanacak_yontem_sayisi = np.random.random_integers(1,3,1)[0]
    grt2 = grt.copy()
    uygulanan_yontemler = []
    for i in range(uygulanacak_yontem_sayisi):
        yontem = yontem_don(uygulanacak_yontem_sayisi)

        if yontem[i] == 1:
            grt2 = yon1(grt2)
            uygulanan_yontemler.append("yon1")
        elif yontem[i] == 2:
            grt2 = yon2(grt2)
            uygulanan_yontemler.append("yon2")
        elif yontem[i] == 3:
            grt2 = yon3(grt2)
            uygulanan_yontemler.append("yon3")
        elif yontem[i] == 4:
            grt2 = yon4(grt2)
            uygulanan_yontemler.append("yon4")
        elif yontem[i] == 5:
            grt2 = yon5(grt2)
            uygulanan_yontemler.append("yon5")
        elif yontem[i] == 6:
            grt2 = yon6(grt2)
            uygulanan_yontemler.append("yon6")

    return grt2,uygulanan_yontemler

def yontemleri_uygula_coklu(grt):
    yontemler = []
    goruntuler = []
    goruntu_sayisi = len(grt)

    for j in range(goruntu_sayisi):
        uygulanacak_yontem_sayisi = np.random.random_integers(1,3,1)[0]
        grt2 = grt[j]
        uygulanan_yontemler = []

        for i in range(uygulanacak_yontem_sayisi):
            yontem = yontem_don(uygulanacak_yontem_sayisi)

            if yontem[i] == 1:
                grt2 = yon1(grt2)
                uygulanan_yontemler.append("yon1")
            elif yontem[i] == 2:
                grt2 = yon2(grt2)
                uygulanan_yontemler.append("yon2")
            elif yontem[i] == 3:
                grt2 = yon3(grt2)
                uygulanan_yontemler.append("yon3")
            elif yontem[i] == 4:
                grt2 = yon4(grt2)
                uygulanan_yontemler.append("yon4")
            elif yontem[i] == 5:
                grt2 = yon5(grt2)
                uygulanan_yontemler.append("yon5")
            elif yontem[i] == 6:
                grt2 = yon6(grt2)
                uygulanan_yontemler.append("yon6")

        yontemler.append(uygulanan_yontemler)
        goruntuler.append(grt2)

    return goruntuler,yontemler
