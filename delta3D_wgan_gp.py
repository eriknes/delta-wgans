from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


import keras.backend as K
import keras.models as kmod
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.merge import _Merge
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras import initializers
from functools import partial
#import loadData3D as d3d

LATENT_VEC_SIZE         = 20
BATCH_COUNT             = 5
BATCH_SIZE              = 36
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC_ITER           = 5
ADAM_LR                 = .0002
ADAM_BETA_1             = 0.5
ADAM_BETA_2             = 0.9

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
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class wGAN():
    def __init__(self, nx, ny, nz, nchan):
        
        # Uncomment for deterministic output.
        #np.random.seed(1)

        # Theano uses ordering nchannels, nx, ny, nz
        K.set_image_dim_ordering('th')
        #(nchan,nx,ny,nz)        = X_train[0].shape
        self.nrows              = nx
        self.ncols              = ny
        self.nlayers            = nz
        self.nchan              = nchan
        self.image_dimensions   = (nchan, self.nrows, self.ncols, self.nlayers)
        print("Image dim is: " )
        print( self.image_dimensions)
        
        self.batch_size         = BATCH_SIZE
        self.latent_dim         = LATENT_VEC_SIZE

        optim               = Adam(lr = ADAM_LR, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
        #optim               = Adam(lr = ADAM_LR, beta_1 = ADAM_BETA_1)


        # Build the generator
        self.generator      = self.buildGenerator()
        # Build discriminator
        self.discriminator  = self.buildDiscriminator()
        # Set trainable = false for the discriminator layers in full model
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False

        # The generator takes noise as input and generated imgs

        generator_input             = Input(shape=(self.latent_dim,))
        generator_layers            = self.generator(generator_input)
        discriminator_layers        = self.discriminator(generator_layers)
        self.generator_model        = Model(inputs=[generator_input], outputs=[discriminator_layers])
        self.generator_model.compile(optimizer = optim, loss = wassersteinLoss)

        # After generator model compilation, we make the discriminator layers trainable.
        for layer in self.discriminator.layers:
            layer.trainable                 = True
        for layer in self.generator.layers:
            layer.trainable                 = False
        self.discriminator.trainable        = True
        self.generator.trainable            = False


        real_samples                            = Input(shape=(self.nchan, nx, ny, nz))
        generator_input_for_discriminator       = Input(shape=(self.latent_dim,))
        generated_samples_for_discriminator     = self.generator(generator_input_for_discriminator)
        discriminator_output_from_generator     = self.discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples  = self.discriminator(real_samples)

        # We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
        averaged_samples        = RandomWeightedAverage()([ real_samples, generated_samples_for_discriminator])

        # We then run these samples through the discriminator as well. Note that we never really use the discriminator
        # output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
        averaged_samples_out    = self.discriminator(averaged_samples)

        # The gradient penalty loss function requires the input averaged samples to get gradients. However,
        # Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
        # of the function with the averaged samples here.
        partial_gp_loss         = partial(gradientPenaltyLoss, averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names in Keras

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

        generator = Sequential()
        generator.add(Dense(32*12*12*3, input_dim=self.latent_dim, 
            kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        generator.add(Activation("relu"))
        #generator.add(Dropout(0.2))
        generator.add(Reshape((32, 12, 12, 3)))
        generator.add(UpSampling3D(size=(2,2,2)))
        generator.add(Conv3D(64, kernel_size=(5, 5, 5), padding='same'))
        generator.add(Activation("relu"))
        generator.add(UpSampling3D(size=(2, 2, 2)))
        generator.add(Conv3D(128, kernel_size=(5, 5, 5), padding='same'))
        generator.add(Activation("relu"))
        generator.add(UpSampling3D(size=(2, 2, 2)))
        generator.add(Conv3D(self.nchan, kernel_size=(5, 5, 5), padding='same', activation='sigmoid'))
        generator.summary()

        return generator

    def buildDiscriminator(self):

        discriminator = Sequential()

        discriminator.add(Conv3D(32, kernel_size=(5,5,5), strides=(2,2,2), input_shape=self.image_dimensions, 
            padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Conv3D(64, kernel_size=(5,5,5), strides=(2,2,2), padding="same"))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Conv3D(128, kernel_size=(5,5,3), strides=(2,2,2), padding="same"))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())

        discriminator.add(Dense(1))

        discriminator.summary()

        return discriminator

    def trainGAN(self, generator, n_epochs = 10, batch_size = 64, sample_interval = 1):
        
        # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
        # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
        # gradient_penalty loss function and is not used.
        positive_y  = np.ones((self.batch_size, 1), dtype=np.float32)
        negative_y  = - positive_y
        dummy_y     = np.zeros((self.batch_size, 1), dtype=np.float32)


        eps = .3
        randomDim = 20

        #batch_count = int(X_train.shape[0] / (self.batch_size * N_CRITIC_ITER))
        #minibatch_size = int(batch_count * N_CRITIC_ITER)

        dLosses                     = []
        gLosses                     = []

        for epoch in range(n_epochs + 1):
            print ("Start epoch %d ----------" % epoch)
            # shuffle Xtrain
            #np.random.shuffle(X_train)
            #print("Epoch: ", epoch)
            #print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))

            nSamples        = BATCH_COUNT*N_CRITIC_ITER
            noise           = np.random.normal(0, 1, size=[nSamples, randomDim])
            generatedCube   = np.zeros((nSamples, self.nrows,self.ncols, self.nlayers))
            generatedImages = generator.predict(noise)

            generatedCube[:,:,:,0] = np.round(np.reshape(generatedImages, (nSamples, self.nrows, self.ncols)))

            # Create cube

            for i in range(1,self.nlayers):
              noise2            = np.random.normal(0, 1, size=[nSamples, randomDim])
              noise             = noise + eps*noise2
              generatedImages   = generator.predict(noise)
              newLayer          = np.reshape(generatedImages, (nSamples, self.nrows, self.ncols))
              generatedCube[:,:,:,i] = np.round(newLayer)

            # Insert channel dimension 
            generatedCube                     = generatedCube[:, np.newaxis, :, :, :]

            for _ in range(BATCH_COUNT):

                #discriminator_minibatches = X_train[i * minibatch_size:(i + 1) * minibatch_size]

                for j in range(N_CRITIC_ITER):

                    # ---------------------
                    #  1 Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx         = np.random.randint(0, generatedCube.shape[0], batch_size)
                    image_batch = generatedCube[idx]

                    # Sample noise as generator input
                    noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim]).astype(np.float32)

                    # Train discriminator model
                    #print(image_batch.shape)
                    #print(noise.shape)
                    #print(negative_y.shape)
                    #print(positive_y.shape)
                    #print(dummy_y.shape)
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
            print ("Epoch %d, [D loss: %f] [G loss: %f]" % (epoch, .5*(d_loss[0] + d_loss[1]), g_loss))
                    
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.saveGenImages(epoch)
                self.saveSampleData(epoch, generatedCube)
                self.saveModels(epoch)
                self.plotLoss(epoch, dLosses, gLosses)

    def saveSampleData(self, epoch, cube, examples=24, dim=(5, 5), figsize=(10, 10)):
        
        #noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
        #generated_images = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(examples):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(cube[0, 0, :, :, i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
            plt.title('Layer %d' % (i+1) )
        plt.tight_layout()
        plt.savefig('images/training_data_epoch_%d.png' % epoch)
        plt.close()

        # Plot the loss from each batch
    def plotLoss(self, epoch, dLosses, gLosses):
        plt.figure(figsize=(10, 10))
        plt.plot(dLosses, label='Discriminator loss')
        plt.plot(gLosses, label='Generator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/loss_epoch_%d.png' % epoch)
        plt.close()

    def saveGenImages(self, epoch, examples=24, dim=(5, 5), figsize=(10, 10)):
        
        noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
        generated_images = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(examples):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[0, 0, :, :, i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
            plt.title('Layer %d' % (i+1) )
        plt.tight_layout()
        plt.savefig('images/wgan_image_epoch_%d.png' % epoch)
        plt.close()


    # Save the generator and discriminator networks (and weights) for later use
    def saveModels(self, epoch):
        self.generator.save('models/wgan3D_gen_ep_%d.h5' % epoch)
        #self.discriminator.save('models/wgan_discriminator_epoch_%d.h5' % epoch)

def buildDataset_3D(filename, datatype='uint8', nx=96, ny=96, nz=16):

  X                     = pd.read_csv(filename, header=None) 
  X                     = X.values.astype(datatype)
  X                     = X.T
  m                     = X.shape[0]
  n                     = X.shape[1]

  print("Number of images: " + str(m) )
  
  if (n != nx*ny*nz):
    print("The number of rows is incorrect")
    exit()

  
  
  # Random permutation of samples
  p         = np.random.permutation(m)
  X         = X[p,:]
  
  # Reshape X and crop to 96x96 pixels
  X_train = np.zeros((m,nx,ny,nz))

  for i in range(m):
    Xtemp = np.reshape(X[i,:],(nz,nx,ny))
    X_train[i,:,:,:] = np.moveaxis(Xtemp, 0, -1)

  print("X_train shape: " + str(X_train.shape))
  
  return X_train

if __name__ == '__main__':
    
    # Load dataset
    generator           = kmod.load_model('models/braided_gen_latent20.h5', 
            custom_objects={'wassersteinLoss': wassersteinLoss})
    #filename                    = "data/train/test3D.csv"
    datatype                    = 'uint8'
    nx                          = 96
    ny                          = 96
    nz                          = 24
    nchan                       = 1

    #X_train                     = buildDataset_3D(filename, datatype, nx, ny, nz)
    
    # Insert channel dimension 
    #X_train                     = X_train[:, np.newaxis, :, :, :]

    # Initialize a class instance
    wgan                        = wGAN(nx, ny, nz, nchan)
    # Start training
    wgan.trainGAN(generator, n_epochs = 500, batch_size = BATCH_SIZE, sample_interval = 5)

