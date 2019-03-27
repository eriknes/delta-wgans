from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.merge import _Merge
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Convolution2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras import initializers
from functools import partial

BATCH_SIZE              = 64
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC_ITER           = 5

def wassersteinLoss(y_true, y_pred):
    """Wasserstein loss"""
    return K.mean(y_true * y_pred)

def gradientPenaltyLoss(y_true, y_pred, averaged_samples, gradient_penalty_weight):

    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # mean loss over samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class wGAN():
    def __init__(self, X_train):

        #np.random.seed(1)

        K.set_image_dim_ordering('th')
        self.nrows                  = X_train.shape[2]
        self.ncols                  = X_train.shape[3]
        self.nchan                  = X_train.shape[1]
        self.image_dimensions       = (self.nchan, self.nrows, self.ncols)
        
        self.batch_size             = BATCH_SIZE
        self.latent_dim             = 16

        # Adam gradient descent
        #optim               = Adam(lr = 0.0001, beta_1 = 0.5, beta_2 = 0.99)
        optim               = Adam(lr = 0.0001, beta_1 = 0.5, beta_2 = 0.99)

        # Build the generator
        self.generator      = self.buildGenerator()
        # Build discriminator
        self.discriminator  = self.buildDiscriminator()
        # Set trainable = false for the discriminator layers in full model
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False

        generator_input     = Input(shape=(self.latent_dim,))
        generator_layers    = self.generator(generator_input)
        discriminator_layers= self.discriminator(generator_layers)
        self.generator_model     = Model(inputs=[generator_input], outputs=[discriminator_layers])
        self.generator_model.compile(optimizer = optim, loss = wassersteinLoss)

        # After generator model compilation, we make the discriminator layers trainable.
        for layer in self.discriminator.layers:
            layer.trainable = True
        for layer in self.generator.layers:
            layer.trainable = False
        self.discriminator.trainable    = True
        self.generator.trainable        = False


        real_samples                        = Input(shape=X_train.shape[1:])
        generator_input_for_discriminator   = Input(shape=(self.latent_dim,))
        generated_samples_for_discriminator = self.generator(generator_input_for_discriminator)
        discriminator_output_from_generator = self.discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples = self.discriminator(real_samples)

        # average of real and GAN
        averaged_samples = RandomWeightedAverage()([ real_samples, generated_samples_for_discriminator])

        averaged_samples_out = self.discriminator(averaged_samples)

        partial_gp_loss = partial(gradientPenaltyLoss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  

        self.discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])

        self.discriminator_model.compile(optimizer= optim,
                                    loss=[wassersteinLoss,
                                          wassersteinLoss,
                                          partial_gp_loss])

    def buildGenerator(self):

        generator = Sequential()
        generator.add(Dense(256*12*12, input_dim=self.latent_dim, 
            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        #generator.add(LeakyReLU(.2))
        generator.add(Activation("relu"))
        #generator.add(Dropout(0.2))
        generator.add(Reshape((256, 12, 12)))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(128, kernel_size=(5,5), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        generator.add(Activation("relu"))
        
        generator.add(Conv2D(self.nchan, kernel_size=(5, 5), padding='same', activation='sigmoid',
            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.summary()

        return generator


    def buildDiscriminator(self):

        discriminator = Sequential()

        discriminator.add(Convolution2D(256, kernel_size=(5,5), strides=(2,2), input_shape=self.image_dimensions, 
            padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(128, kernel_size=(5,5), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(128, kernel_size=(5,5), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(128, kernel_size=(5,5), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(64, kernel_size=(5,5), strides=(2,2), padding="same"))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())
        
        discriminator.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

        discriminator.summary()

        return discriminator

    def trainGAN(self, X_train, epochs = 10, batch_size = 64, sample_interval = 1):


        positive_y  = np.ones((self.batch_size, 1), dtype=np.float32)
        negative_y  = - positive_y
        dummy_y     = np.zeros((self.batch_size, 1), dtype=np.float32)

        batch_count = 1

        dLosses                     = []
        gLosses                     = []

        for epoch in range(epochs + 1):
            # shuffle Xtrain
            np.random.shuffle(X_train)

            for i in range(batch_count):

                for j in range(N_CRITIC_ITER):

                    # ---------------------
                    #  1 Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    image_batch = X_train[idx]
                    #image_batch = discriminator_minibatches[j*batch_size:(j+1)*batch_size]

                    # Sample noise as generator input
                    noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim]).astype(np.float32)

                    # Generate a batch of new images
                    #gen_images = self.generator.predict(noise)
                    d_loss = self.discriminator_model.train_on_batch([image_batch, noise],
                                                                 [positive_y, negative_y, dummy_y])


                # ---------------------
                #  2 Train Generator
                # ---------------------
                noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim]).astype(np.float32)
                g_loss = self.generator_model.train_on_batch(noise, positive_y)
            gLosses.append(g_loss)
            dLosses.append(-d_loss[0] - d_loss[1] - d_loss[2])
                
            # Print the progress
            if epoch % (sample_interval/10) == 0:
                print ("Iteration %d, [D loss: %f] [G loss: %f]" % (epoch, .5*(d_loss[0] + d_loss[1]), g_loss))
                    
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.plotGeneratedImages(epoch, 4, (4, self.nchan))
                self.plotSampleImages(epoch, image_batch, 4, (4, self.nchan))
                self.saveModels(epoch)
                self.plotLoss(epoch, dLosses, gLosses)

    def plotGeneratedImages(self, epoch, examples=4, dim=(4,2), figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
        generated_images = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(examples):
            for j in range(self.nchan):
                plt.subplot(dim[0], dim[1], i*self.nchan+j+1)
                plt.imshow(generated_images[i, j], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                plt.title("Channel " + str(j))
        plt.tight_layout()
        plt.savefig('images/wgan_image_iter_%d.png' % epoch)
        plt.close()

    def plotSampleImages(self, epoch, images, examples=4, dim=(4, 2), figsize=(10, 10)):

        plt.figure(figsize=figsize)
        for i in range(examples):
            for j in range(self.nchan):
                plt.subplot(dim[0], dim[1], i*self.nchan+j+1)
                plt.imshow(images[i, j], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                plt.title("Channel " + str(j))
        plt.tight_layout()
        plt.savefig('images/training_samples_iter_%d.png' % epoch)
        plt.close()

    # Plot the loss from each batch
    def plotLoss(self, epoch, dLosses, gLosses):
        plt.figure(figsize=(10, 8))
        plt.plot(dLosses, label='Discriminitive loss')
        plt.plot(gLosses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/wgan_loss_iter_%d.png' % epoch)
        plt.close()


    def saveModels(self, epoch):
        self.generator.save('models/generator_iter_%d.h5' % epoch)
        self.discriminator.save('models/discriminator_iter_%d.h5' % epoch)

# Read csv file
def load_file(fname):
     X = pd.read_csv(fname)
     X = X.values
     X = X.astype('uint8')
     return X

 # Split into train and test data for GAN 
def build_dataset( filename, nx, ny):

    X           = load_file(filename)

    m           = X.shape[0]
    print("Number of images in dataset: " + str(m) )

    X           = X.T
    #Y = np.zeros((m,))

    # Random permutation of samples
    p           = np.random.permutation(m)
    X           = X[:,p]
    #Y = Y[p]
    nchan       = np.size(np.unique(X))
    print("Number of facies in dataset: " + str(nchan-1))

    # Reshape X and crop to 96x96 pixels
    X_new       = np.zeros((m,nchan-1,nx,ny), dtype='uint8')

    for i in range(m):
        Xtemp1 = np.reshape(X[:,i],(nx,ny)) 
        for j in range(1,nchan):
            Xtemp2              = np.zeros(Xtemp1.shape)
            Xtemp2[np.where(Xtemp1 == j)] = 1
            X_new[i,j-1,:,:]      = Xtemp2

    print("X_train shape: " + str(X_new.shape))
    #print("Y_train shape: " + str(Y_train.shape))

    return X_new

if __name__ == '__main__':
    # Load dataset
    filename                    = "data/train/entireDeltas_3F.csv"
    X_train                     = build_dataset(filename, 192, 192)
    #X_train                     = X_train[:, np.newaxis, :, :]

    wgan = wGAN(X_train)
    wgan.trainGAN(X_train, epochs = 15000, batch_size = BATCH_SIZE, sample_interval = 200)
