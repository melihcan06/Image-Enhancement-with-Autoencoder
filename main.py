from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential,Model
from keras.datasets import mnist
from keras.models import load_model
import numpy as np

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train_noisy = X_train + 0.25 * np.random.normal(size=X_train.shape)
X_test_noisy = X_test + 0.25 * np.random.normal(size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

input_img = Input(shape=(28, 28, 1))

conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1) 
conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2) 
conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2) 

conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3) 
up1 = UpSampling2D((2,2))(conv4)
conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2,2))(conv5) 
conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(up2) 
decoded = Conv2D(1, (3,3), activation='relu', padding='same')(conv6) 

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=64, shuffle=True, validation_split=0.2)

autoencoder.save("model3_4")
#autoencoder=load_model("model3_4")
autoencoder.summary()
