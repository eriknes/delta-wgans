from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import initializers

class wGAN():
	def __init__(self):
		# Deterministic output.
		np.random.seed(1)

		self.nrows 			= 96
		self.ncols 			= 96
		self.nchan 			= 1
		self.dimensions 	= (self.nrows, self.ncols, self.nchan)
		self.latent_dim 	= 10

		self.nCriticIter 	= 5
		self.clip_val 		= 0.1
		self.learning_rate  = 0.0001
		optim 	 			= RMSprop(lr = self.learning_rate, clipvalue = self.clip_val)


		# Build discriminator
		self.discriminator  = self.buildDiscriminator()
		self.discriminator.compile(loss = self.wLoss, optimizer = optim, metrics=['accuracy'])

       	# Build the generator
		self.generator 		= self.buildGenerator()
		# The generator takes noise as input and generated imgs
		genInput 			= Input(shape=(self.latent_dim,))
		img 				= self.generator(genInput)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The critic takes generated images as input and determines validity
		discOutput 			= self.discriminator(img)

		# The combined model  (stacked generator and critic)
		self.combined 		= Model(inputs = genInput, outputs = discOutput)
		self.combined.compile(loss = self.wLoss, optimizer=optim, metrics=['accuracy'])

	def wLoss(self, yReal, yPred):
		return K.mean(yReal*yPred)

	def buildGenerator(self):

		generator = Sequential()
		generator.add(Dense(256 * 12 * 12, input_dim=self.latent_dim),
							kernel_initializer=initializers.RandomNormal(stddev=0.02))
		generator.add(Activation('relu'))
		generator.add(Reshape((256, 12, 12)))
		generator.add(UpSampling2D())
		generator.add(Conv2D(128, kernel_size=(6,6), padding="same", activation = "relu"))
		#generator.add(BatchNormalization(momentum=0.8))
		generator.add(UpSampling2D())
		generator.add(Conv2D(64, kernel_size=(6,6), padding="same", activation = "relu"))
		#generator.add(BatchNormalization(momentum=0.8))
		generator.add(UpSampling2D())
		#generator.add(Conv2D(32, kernel_size=(5,5), padding="same", activation = "relu"))
		generator.add(Conv2D(1, kernel_size=(6,6), padding="same", activation = "sigmoid"))

		generator.summary()

		noise 	= Input(shape=(self.latent_dim,))
		img 	= generator(noise)

		return Model(noise, img)

	def buildDiscriminator(self):

		discriminator = Sequential()

		discriminator.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.dimensions, padding="same"))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.25))
		discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
		discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.25))
		discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.25))
		discriminator.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.25))
		discriminator.add(Flatten())
		discriminator.add(Dense(1))

		discriminator.summary()

		img = Input(shape=self.dimensions)
		validity = discriminator(img)

		return Model(img, validity)

	def trainGAN(self, epochs=1, batch_size=128, sample_interval=2):

		# Load dataset
		filename                   	= "data/train/braidedData2.csv"
		X_train                   	= load_file(filename)
		(X_train, Y_train) 			= build_dataset(X_train, self.nrows, self.ncols)
		X_train 					= np.expand_dims(X_train, axis=3)

		#batch_count 				= X_train.shape[0] / batch_size
		batch_count 				= 1

		# Labels
		y_real 					= np.zeros((2*batch_size, 1))
		# one-sided label smoothing
		y_real[:batch_size] 	= .99*np.ones((batch_size, 1))

		for epoch in range(epochs+1):

			#for _ in xrange(batchCount):

			for _ in range(self.nCriticIter):

				# ---------------------
				#  1 Train Discriminator
				# ---------------------

				# Select a random batch of images
				idx = np.random.randint(0, X_train.shape[0], batch_size)
				image_batch = X_train[idx]

				# Sample noise as generator input
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

				# Generate a batch of new images
				gen_images = self.generator.predict(noise)

				# Train the critic
				print(noise.shape)
				print(image_batch.shape)
				print(gen_images.shape)
				X = np.concatenate([image_batch, gen_images])
				d_loss = self.discriminator.train_on_batch(X, valid)
				#d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
				#d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

				# Clip critic weights
				for l in self.discriminator.layers:
					weights = l.get_weights()
					weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
					l.set_weights(weights)


			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, y_real)

			# Print the progress
			print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.plotGeneratedImages(epoch)
				self.saveModels(epoch)

	def plotGeneratedImages(self, epoch, examples=25, dim=(5, 5), figsize=(10, 10)):
		noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
		generated_images = self.generator.predict(noise)

		plt.figure(figsize=figsize)
		for i in range(examples):
			plt.subplot(dim[0], dim[1], i+1)
			plt.imshow(generated_images[i, 0], interpolation='nearest', cmap='gray_r')
			plt.axis('off')
		plt.tight_layout()
		plt.savefig('images/wgan_image_epoch_%d.png' % epoch)
		plt.close()

	# Save the generator and discriminator networks (and weights) for later use
	def saveModels(self, epoch):
		self.generator.save('models/wgan_generator_epoch_%d.h5' % epoch)
		#self.discriminator.save('models/wgan_discriminator_epoch_%d.h5' % epoch)

# Read csv file
def load_file(fname):
	 X = pd.read_csv(fname)
	 X = X.values
	 X = X.astype('uint8')
	 return X

 # Split into train and test data for GAN 
def build_dataset(X, nx, ny, n_test = 0):

	m = X.shape[0]
	print("Number of images in dataset: " + str(m) )

	X = X.T
	Y = np.zeros((m,))

	# Random permutation of samples
	p = np.random.permutation(m)
	X = X[:,p]
	Y = Y[p]

	# Reshape X and crop to 96x96 pixels
	X_new = np.zeros((m,nx,ny))

	for i in range(m):
		Xtemp = np.reshape(X[:,i],(101,101))
		X_new[i,:,:] = Xtemp[2:98,2:98]

	X_train = X_new[0:m-n_test,:,:]
	Y_train = Y[0:m-n_test]

	#X_test  = X_new[m-n_test:m,:,:]
	#Y_test  = Y[m-n_test:m]

	print("X_train shape: " + str(X_train.shape))
	print("Y_train shape: " + str(Y_train.shape))

	return X_train, Y_train

if __name__ == '__main__':
	wgan = wGAN()
	wgan.trainGAN(epochs = 100, batch_size = 128, sample_interval = 50)
