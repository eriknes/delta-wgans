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
    """Wasserstein loss for a sample batch."""
    return K.mean(y_true * y_pred)

def gradientPenaltyLoss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples."""
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points. Inheritance from _Merge """

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class wGAN():
    def __init__(self, X_train):
        # Deterministic output.
        #np.random.seed(1)

        K.set_image_dim_ordering('th')
        self.nrows          = 96
        self.ncols          = 96
        self.nchan          = 1
        self.image_dimensions     = (self.nchan, self.nrows, self.ncols)
        
        self.batch_size     = BATCH_SIZE
        self.latent_dim     = 20

        #self.nCriticIter    = 5
        #self.clip_val       = 0.01

        optim               = Adam(lr = 0.00005, beta_1 = 0.5, beta_2 = 0.9)


        # Build the generator
        self.generator      = self.buildGeneratorConv()
        # Build discriminator
        self.discriminator  = self.buildDiscriminator()
        # Set trainable = false for the discriminator layers in full model
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False

        # The generator takes noise as input and generated imgs

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

        # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
        averaged_samples = RandomWeightedAverage()([ real_samples, generated_samples_for_discriminator])

        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out = self.discriminator(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get gradients. However,
        # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
        # of the function with the averaged samples here.
        partial_gp_loss = partial(gradientPenaltyLoss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

        # If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
        # samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
        self.discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
        # samples, and the gradient penalty loss for the averaged samples.
        self.discriminator_model.compile(optimizer= optim,
                                    loss=[wassersteinLoss,
                                          wassersteinLoss,
                                          partial_gp_loss])

    def buildGenerator(self):

        bn_axis = 1

        generator = Sequential()
        generator.add(Dense(1024, input_dim=self.latent_dim))
        generator.add(LeakyReLU())

        #generator.add(Activation("relu"))
        
        generator.add(Dense(64*12*12, input_dim=self.latent_dim))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Reshape((64, 12, 12), input_shape=(128 * 12 * 12,)))

        # 24 x 24
        generator.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(128, (5, 5), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        # 48 x 48
        generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(64, (5, 5), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        # 96 x 96
        generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(64, (5, 5), padding='same'))
        generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        generator.add(Conv2D(self.nchan, kernel_size=(5, 5), padding='same', activation='sigmoid'))
        generator.summary()

        return generator

    def buildGeneratorConv(self):

        bn_axis = 1

        generator = Sequential()
        generator.add(Dense(1024, input_dim=self.latent_dim))
        generator.add(LeakyReLU())

        #generator.add(Activation("relu"))
        
        generator.add(Dense(64*12*12, input_dim=self.latent_dim))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Reshape((64, 12, 12), input_shape=(128 * 12 * 12,)))

        # 24 x 24
        generator.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(128, (5, 5), padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        # 48 x 48
        generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(64, (5, 5), padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        # 96 x 96
        generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())
        generator.add(Convolution2D(64, (5, 5), padding='same'))
        #generator.add(BatchNormalization())
        generator.add(LeakyReLU())

        generator.add(Conv2D(self.nchan, kernel_size=(5, 5), padding='same', activation='sigmoid'))
        generator.summary()

        return generator

    def buildDiscriminator(self):

        discriminator = Sequential()

        discriminator.add(Convolution2D(64, kernel_size=(5,5), strides=2, input_shape=self.image_dimensions, 
            padding="same"))
        discriminator.add(LeakyReLU())
        #discriminator.add(Dropout(0.2))

        discriminator.add(Convolution2D(64, kernel_size=(5,5), strides=2, padding="same",
            kernel_initializer='he_normal'))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU())
        #discriminator.add(Dropout(0.2))

        discriminator.add(Convolution2D(128, kernel_size=(5,5), strides=2, padding="same",
            kernel_initializer='he_normal'))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU())
        #discriminator.add(Dropout(0.2))

        discriminator.add(Convolution2D(128, kernel_size=(5,5), strides=2, padding="same",
            kernel_initializer='he_normal'))
        #discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU())
        #discriminator.add(Dropout(0.2))
        discriminator.add(Flatten())

        discriminator.add(Dense(1024, kernel_initializer='he_normal'))
        #discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU())
        #discriminator.add(Dropout(0.2))
        
        discriminator.add(Dense(1, kernel_initializer='he_normal'))

        discriminator.summary()

        return discriminator

    def trainGAN(self, X_train, epochs = 10, batch_size = 64, sample_interval = 1):

        
        # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
        # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
        # gradient_penalty loss function and is not used.
        positive_y  = np.ones((self.batch_size, 1), dtype=np.float32)
        negative_y  = - positive_y
        dummy_y     = np.zeros((self.batch_size, 1), dtype=np.float32)

        batch_count = 10
        #batch_count = int(X_train.shape[0] / (self.batch_size * N_CRITIC_ITER))
        #minibatch_size = int(batch_count * N_CRITIC_ITER)

        dLosses                     = []
        gLosses                     = []

        for epoch in range(epochs + 1):
            # shuffle Xtrain
            np.random.shuffle(X_train)
            #print("Epoch: ", epoch)
            #print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))

            for i in range(batch_count):

                #discriminator_minibatches = X_train[i * minibatch_size:(i + 1) * minibatch_size]

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
                dLosses.append(.5*(d_loss[0] + d_loss[1]))
                
                # Print the progress
                if i % 10 == 0:
                    print ("Epoch %d, batch %d: [D loss: %f] [G loss: %f]" % (epoch, i, .5*(d_loss[0] + d_loss[1]), g_loss))

            
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.plotGeneratedImages(epoch)
                #self.plotSampleImages(epoch, image_batch)
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
def build_dataset( filename, nx, ny, n_test = 0):

    X                     = load_file(filename)

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
    # Load dataset
    filename                    = "data/train/braidedData2.csv"
    (X_train, y_train) = build_dataset(filename, 96, 96, 0)
    X_train                     = X_train[:, np.newaxis, :, :]

    wgan = wGAN(X_train)
    wgan.trainGAN(X_train, epochs = 50, batch_size = BATCH_SIZE, sample_interval = 1)
