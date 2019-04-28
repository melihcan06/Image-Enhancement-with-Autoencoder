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

autoencoder = Sequential()
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding ='same', input_shape = (28, 28, 1)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding ='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding ='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding ='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))



autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding ='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding ='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3, 3), activation='relu', ))
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding ='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3, 3), activation='relu', padding ='same'))


autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
autoencoder.fit(X_train_noisy, X_train, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

autoencoder.save("model3_3")
#autoencoder=load_model("model3_3")
autoencoder.summary()
