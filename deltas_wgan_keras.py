from __future__ import print_function, division

import numpy as np
import pandas as pd
import tqdm
import sys
import matplotlib.pyplot as plt
import keras.backend as K


from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from deltaData import *

class wGAN():
	def __init__(self):
		# Deterministic output.
		np.random.seed(1)

		self.nrows 			= 96
		self.ncols 			= 96
		self.nchan 			= 1
		self.dimensions 	= (self.nrows, self.ncols, self.nchan)
		self.latent_dim 	= 20

		self.nCriticIter 	= 5
		self.clip_val 		= 0.1
		self.learning_rate  = 0.0001
		optim 	 			= RMSprop(lr = self.learning_rate, clipvalue = self.clip_val)


		# Build discriminator
		self.discriminator  = self.buildDiscriminator()
		self.discriminator.compile(loss = self.wLoss, optimizer = optim)

       	# Build the generator
        self.generator = self.buildGenerator()
		# The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wLoss,
            optimizer=optim)

	def wLoss(self, yReal, yPred):
		return K.mean(yReal*yPred)

    def buildGenerator(self):

        generator = Sequential()

        generator.add(Dense(128 * 12 * 12, activation="relu", input_dim=self.latent_dim))
        generator.add(Reshape((128, 12, 12)))
        generator.add(UpSampling2D(size=(2,2)))
        generator.add(Conv2D(128, kernel_size=5, padding="same"), activation = "relu")
        #generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D(size=(2,2)))
        generator.add(Conv2D(64, kernel_size=5, padding="same"), activation = "relu")
        #generator.add(BatchNormalization(momentum=0.8))
        generator.add(UpSampling2D(size=(2,2)))
        generator.add(Conv2D(32, kernel_size=5, padding="same"), activation = "relu")
        generator.add(Conv2D(self.nchan, kernel_size=5, padding="same"), activation = "sigmoid")

        generator.summary()

        noise 	= Input(shape=(self.latent_dim,))
        img 	= generator(noise)

        return Model(noise, img)

    def buildDiscriminator(self):

    	discriminator = Sequential()

        discriminator.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
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
        discriminator.add(Dense(1), activation = 'sigmoid')

        discriminator.summary()

        img = Input(shape=self.img_shape)
        validity = discriminator(img)

        return Model(img, validity)

    def trainGAN(self, epochs=1, batch_size=128, sample_interval=2):

        # Load dataset
		filename                   	= "data/train/braidedData2.csv"
		X_train                   	= load_file(filename)
        (X_train, _), (_, _) 		= build_dataset(X_train, self.nrows, self.ncols)

        batch_count 				= X_train.shape[0] / batch_size

        # Labels
        y_real 					= np.zeros((2*batch_size, 1))
        # one-sided label smoothing
        y_real[:batch_size] 	= .99*np.ones((batch_size, 1))

        for epoch in range(epochs+1):

        	for _ in xrange(batchCount):

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

	            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

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


if __name__ = '__main__':
	wgan = wGAN()
	wgan.trainGAN(epochs = 100, batch_size = 128, sample_interval = 50)