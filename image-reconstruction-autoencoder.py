import tensorflow
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

# build fully connectd feedforward autoencoder
n_codings = 3

stacked_encoder = models.Sequential([
	layers.Flatten(input_shape=[28, 28]),
	layers.Dense(100, activation='selu'),
	layers.Dense(n_codings, activation='selu')
	])

stacked_decoder = models.Sequential([
	layers.Dense(100, activation='selu', input_shape=[n_codings]),
	layers.Dense(28 * 28, activation='sigmoid'),
	layers.Reshape([28, 28])
	])

stacked_ae = models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss='binary_crossentropy', optimizer=SGD(lr=1.5))
history = stacked_ae.fit(X_train, X_train, epochs=10,
							validation_data=(X_test, X_test))

# plot test images and corresponding reconstructions
n_images = 5
reconstructions = stacked_ae.predict(X_test[:n_images])
fig = plt.figure(figsize=(n_images * 1.5, 3))
for i in range(n_images):
	plt.subplot(2, n_images, 1 + i)
	plt.imshow(X_test[i], cmap='binary')
	plt.axis('off')
	plt.subplot(2, n_images, 1 + n_images + i)
	plt.imshow(reconstructions[i], cmap='binary')
	plt.axis('off')
plt.show()
