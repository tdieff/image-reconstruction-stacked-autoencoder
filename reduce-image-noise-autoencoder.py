emport tensorflow
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.optimizers import SGD

# import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# scale data
X_train = X_train / 255
X_test = X_test / 255

# generate noisy data
X_train_noisy = np.clip(X_train
	+ np.random.normal(loc=0.0, scale=0.5, size=X_train.shape), 0., 1.)
X_test_noisy = np.clip(X_test
	+ np.random.normal(loc=0.0, scale=0.5, size=X_test.shape), 0., 1.)

# build convolutional autoencoder
conv_encoder = models.Sequential([
	layers.Reshape([28, 28, 1], input_shape=[28, 28]),
	layers.Conv2D(16, (3, 3), padding='same', activation='selu'),
	layers.MaxPool2D((2, 2)),
	layers.Conv2D(8, (3, 3), padding='same', activation='selu'),
	layers.MaxPool2D((2, 2))
	])

conv_decoder = models.Sequential([
	layers.Conv2D(8, (3, 3), padding='same', activation='selu',
											input_shape=[7, 7, 8]),
	layers.UpSampling2D((2, 2)),
	layers.Conv2D(4, (3, 3), padding='same', activation='selu'),
	layers.UpSampling2D((2, 2)),
	layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'),
	layers.Reshape([28, 28])
	])

conv_ae = models.Sequential([conv_encoder, conv_decoder])
conv_ae.compile(loss='binary_crossentropy', optimizer='adam')
history = conv_ae.fit(X_train_noisy, X_train, epochs=20,
							validation_data=(X_test_noisy, X_test))

# plot test images and corresponding reconstructions
n_images = 5
reconstructions = conv_ae.predict(X_test_noisy[:n_images])
fig = plt.figure(figsize=(n_images * 1.5, 3))
for i in range(n_images):
	plt.subplot(2, n_images, 1 + i)
	plt.imshow(X_test_noisy[i], cmap='binary')
	plt.axis('off')
	plt.subplot(2, n_images, 1 + n_images + i)
	plt.imshow(reconstructions[i], cmap='binary')
	plt.axis('off')
plt.show()


# conv_encoder = models.Sequential([
# 	layers.Conv2D(32, kernel_size=3, padding='same', activation='selu',
# 		input_shape=(28, 28, 1)),
# 	layers.MaxPool2D(pool_size=2, padding='same'),
# 	layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'),
# 	layers.MaxPool2D(pool_size=2, padding='same')
# 	])

# conv_decoder = models.Sequential([
# 	layers.Conv2DTranspose(16, kernel_size=3, padding='same', activation='selu',
# 															input_shape=(7, 7, 16)),
# 	layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='selu'),
# 	layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')
# 	])


# conv_encoder = models.Sequential([
# 	layers.Reshape([28, 28, 1], input_shape=[28, 28]),
# 	layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'),
# 	layers.MaxPool2D(pool_size=2),
# 	layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'),
# 	layers.MaxPool2D(pool_size=2),
# 	layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'),
# 	layers.MaxPool2D(pool_size=2)
# 	])

# conv_decoder = models.Sequential([
# 	layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='valid',
# 							activation='selu', input_shape=[3, 3, 64]),
# 	layers.Conv2DTranspose(16, kernel_size=3, strides=2,
# 							padding='same', activation='selu'),
# 	layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid'),
# 	layers.Reshape([28, 28])
# 	])
