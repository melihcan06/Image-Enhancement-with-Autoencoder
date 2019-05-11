import cv2
import numpy as np
from math import sqrt,log10
import matplotlib.pyplot as plt

def bas(img):
    for i in range(len(img)):
        cv2.imshow("grt"+str(i+1),img[i])

    cv2.waitKey(0)

def ciz(hist):
    renkler=['red','blue','yellow','black','green','orange']
    m=0
    x=np.arange(256)
    for i in range(len(hist)):
        if m == len(renkler):
            m=0

        plt.plot(x,hist[i],c=renkler[m])
        m+=1
    plt.show()

def histHesapla(grt):
    return cv2.calcHist([grt], [0], None, [256], [0, 256])

def cizdir(grtler):
    histler=[]
    for i in grtler:
        histler.append(histHesapla(i))
    ciz(histler)

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)

def gaussian_noise(image_in, noise_sigma=35):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return convert_to_uint8(noisy_image)

def salt_pepper(image):
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out

def gamma(image, gamma=1.0,c=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)])
    table=np.clip(table,0,255).astype("uint8")
    return cv2.LUT(image, table)

def log_transform(grt,c):
    table=np.array([c*log10(1+i)
                    for i in range(0,256)])
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(grt,table)

def blur(grt,blur,kernel_size=(3,3)):
    if blur == 0 or blur == "median":
        return cv2.medianBlur(grt,kernel_size[0])
    elif blur == 1 or blur == "gaussian":
        return cv2.GaussianBlur(grt,kernel_size,0)
    elif blur == 2 or blur == "mean":
        return cv2.blur(grt,kernel_size)

def dogru_hesapla(x1,y1,x2,y2):
    m=(y2-y1)/float(x2-x1)
    n=y2-m*x2
    return m,n

def yeginlik_seviyesi_dilimleme(grt,x1=50,y1=60,x2=225,y2=200,baslangic_y=50):
    tablo=[i for i in range(256)]

    l=256

    m1,n1=dogru_hesapla(0,baslangic_y,x1,y1)
    m2,n2=dogru_hesapla(x1,y1,x2,y2)
    m3,n3=dogru_hesapla(x2,y2,l-1,l-1)

    for i in range(256):
        if tablo[i] < x1:
            tablo[i]=m1*float(tablo[i])+n1

        elif tablo[i] < x2:
            tablo[i] = m2 * float(tablo[i]) + n2

        else:
            tablo[i] = m3 * float(tablo[i]) + n3

    tablo = np.clip(tablo,0,255).astype("uint8")

    return cv2.LUT(grt,tablo)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))
