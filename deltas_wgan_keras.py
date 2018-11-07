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
		#np.random.seed(1)

		self.nrows 			= 96
		self.ncols 			= 96
		self.nchan 			= 1
		self.dimensions 	= (self.nchan, self.nrows, self.ncols)
		self.latent_dim 	= 50

		self.nCriticIter 	= 10
		self.clip_val 		= 0.01
		self.learning_rate  = 0.00005
		optim 	 			= RMSprop(lr = self.learning_rate)


		# Build discriminator
		self.discriminator  = self.buildDiscriminator()
		# For the combined model we will only train the generator
		self.discriminator.trainable = True
		self.discriminator.compile(loss = self.wLoss, optimizer = optim, metrics=['accuracy'])

       	# Build the generator
		self.generator 		= self.buildGenerator()
		# The generator takes noise as input and generated imgs
		genInput 			= Input(shape=(self.latent_dim,))
		img 				= self.generator(genInput)


		# The critic takes generated images as input and determines validity
		discOutput 			= self.discriminator(img)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The combined model  (stacked generator and critic)
		self.combined 		= Model(inputs = genInput, outputs = discOutput)
		self.combined.compile(loss = self.wLoss, optimizer=optim, metrics=['accuracy'])

	def wLoss(self, yReal, yPred):
		return K.mean(yReal*yPred)

	def buildGenerator(self):

		generator = Sequential()
		generator.add(Dense(256*6*6, input_dim=self.latent_dim, 
  			kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		generator.add(Activation('relu'))
		#generator.add(Dropout(0.2))
		generator.add(Reshape((256, 6, 6)))
		generator.add(UpSampling2D(size=(2, 2)))
		generator.add(Conv2D(128, kernel_size=(5,5), padding='same', 
			kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		#generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation('relu'))
		generator.add(UpSampling2D(size=(2, 2)))
		generator.add(Conv2D(128, kernel_size=(5,5), padding='same', 
			kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		#generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation('relu'))
		generator.add(UpSampling2D(size=(2, 2)))
		generator.add(Conv2D(64, kernel_size=(5, 5), padding='same', 
			kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		#generator.add(BatchNormalization(momentum=0.8))
		generator.add(Activation('relu'))
		generator.add(UpSampling2D(size=(2, 2)))
		generator.add(Conv2D(self.nchan, kernel_size=(5, 5), padding='same', activation='sigmoid'))
		generator.summary()

		noise 	= Input(shape=(self.latent_dim,))
		img 	= generator(noise)

		return Model(noise, img)

	def buildDiscriminator(self):

		discriminator = Sequential()

		discriminator.add(Conv2D(32, kernel_size=(5,5), strides=2, input_shape=self.dimensions, 
			padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.02)))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.2))
		discriminator.add(Conv2D(64, kernel_size=(5,5), strides=2, padding="same",
			kernel_initializer='he_normal'))
		#discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.2))
		discriminator.add(Conv2D(128, kernel_size=(5,5), strides=2, padding="same",
			kernel_initializer='he_normal'))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.2))
		discriminator.add(Conv2D(256, kernel_size=(5,5), strides=2, padding="same",
			kernel_initializer='he_normal'))
		discriminator.add(BatchNormalization(momentum=0.8))
		discriminator.add(LeakyReLU(alpha=0.2))
		discriminator.add(Dropout(0.2))
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
		X_train                   	= X_train[:, np.newaxis, :, :]

		#batch_count 				= X_train.shape[0] / batch_size
		batch_count 				= 1

		# Fake = 1 Real = -1
		y_fake 						= np.ones((batch_size, 1))
		# 
		y_real 						= -np.ones((batch_size, 1))

		dLosses                   	= []
		gLosses                   	= []

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
				noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim])

				# Generate a batch of new images
				gen_images = self.generator.predict(noise)

				# Train the critic (do not concatenate images)
				#X = np.concatenate([image_batch, gen_images])
				d_loss_real = self.discriminator.train_on_batch(image_batch, y_real)
				d_loss_fake = self.discriminator.train_on_batch(gen_images, y_fake)
				#d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
				d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

				# Clip critic weights
				for l in self.discriminator.layers:
					weights = l.get_weights()
					weights = [np.clip(w, -self.clip_val, self.clip_val) for w in weights]
					l.set_weights(weights)


			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, y_real)

			dLosses.append(d_loss)
			gLosses.append(g_loss)

			# Print the progress
			print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1-d_loss[0], 1-g_loss[0]))

			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.plotGeneratedImages(epoch)
				self.plotSampleImages(epoch, image_batch)
				self.saveModels(epoch)
				self.plotLoss(epoch, dLosses, gLosses)

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

	def plotSampleImages(self, epoch, images, examples=25, dim=(5, 5), figsize=(10, 10)):

		plt.figure(figsize=figsize)
		for i in range(examples):
			plt.subplot(dim[0], dim[1], i+1)
			plt.imshow(images[i, 0], interpolation='nearest', cmap='gray_r')
			plt.axis('off')
		plt.tight_layout()
		plt.savefig('images/training_samples_epoch_%d.png' % epoch)
		plt.close()

	# Plot the loss from each batch
	def plotLoss(self, epoch, dLosses, gLosses):
		plt.figure(figsize=(10, 8))
		plt.plot(dLosses, label='Discriminitive loss')
		plt.plot(gLosses, label='Generative loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('images/wgan_loss_epoch_%d.png' % epoch)
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
	wgan.trainGAN(epochs = 1000, batch_size = 64, sample_interval = 50)
