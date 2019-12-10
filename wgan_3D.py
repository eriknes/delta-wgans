from __future__ import print_function, division

import sys
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display variable found - using non-interactive agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import keras.backend as K
import keras.models as kmod
import sys
import functools


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

SAMPLE_INT              = 500
EPSILON                 = 0.2
LATENT_VEC_SIZE         = 32
BATCH_SIZE              = 32
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC_ITER           = 5
NUM_ITER                = 15000
ADAM_LR                 = .0001
ADAM_BETA_1             = 0.5
ADAM_BETA_2             = 0.9

def wassersteinLoss(y_true, y_pred):
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
    def __init__(self, nx, ny, nz, nchan, max_filters, kernel_sz):
        

        K.set_image_dim_ordering('th')
        #(nchan,nx,ny,nz)        = X_train[0].shape
        self.nrows              = nx
        self.ncols              = ny
        self.nlayers            = nz
        self.nchan              = nchan
        self.image_dimensions   = (self.nchan, self.nrows, self.ncols, self.nlayers)
        print("Image dim is: " )
        print( self.image_dimensions)
        
        self.batch_size         = BATCH_SIZE
        self.latent_dim         = LATENT_VEC_SIZE

        #optim               = Adam(lr = ADAM_LR, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
        optim               = Adam(lr = ADAM_LR, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)


        # Build the generator
        self.generator      = self.buildGenerator(max_filters, kernel_sz)
        # Build discriminator
        self.discriminator  = self.buildDiscriminator(max_filters, kernel_sz)
        # Set trainable = false for the discriminator layers in full model
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.discriminator.trainable = False

        generator_input             = Input(shape=(self.latent_dim,))
        generator_layers            = self.generator(generator_input)
        discriminator_layers        = self.discriminator(generator_layers)
        self.generator_model        = Model(inputs=[generator_input], outputs=[discriminator_layers])
        self.generator_model.compile(optimizer = optim, loss = wassersteinLoss)

        # After generator model compilation, make discriminator layers trainable.
        for layer in self.discriminator.layers:
            layer.trainable                 = True
        for layer in self.generator.layers:
            layer.trainable                 = False
        self.discriminator.trainable        = True
        self.generator.trainable            = False

        real_samples                            = Input(shape=self.image_dimensions)
        generator_input_for_discriminator       = Input(shape=(self.latent_dim,))
        generated_samples_for_discriminator     = self.generator(generator_input_for_discriminator)
        discriminator_output_from_generator     = self.discriminator(generated_samples_for_discriminator)
        discriminator_output_from_real_samples  = self.discriminator(real_samples)
        print("Real samples shape is: " )
        print(real_samples.shape)
        print("Generated samples shape is: " )
        print(generated_samples_for_discriminator.shape)
        averaged_samples        = RandomWeightedAverage()([ real_samples, generated_samples_for_discriminator])

        averaged_samples_out    = self.discriminator(averaged_samples)

        partial_gp_loss         = partial(gradientPenaltyLoss, averaged_samples=averaged_samples,
                                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names in Keras

        self.discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])
        
        # Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
        # samples, gradient penalty loss for the averaged samples.
        self.discriminator_model.compile(optimizer= optim,
                                    loss=[wassersteinLoss,
                                          wassersteinLoss,
                                          partial_gp_loss])

    def buildGenerator(self, max_filters, kernel_sz):

        generator = Sequential()
        #generator.add(Dense(128, input_dim=self.latent_dim, 
        #    kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        #generator.add(Activation("relu"))
        generator.add(Dense(128*12*12*3, input_dim=self.latent_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(Reshape((128, 12, 12, 3)))

        generator.add(Activation("relu"))
        generator.add(UpSampling3D(size=(2, 2, 2)))
        #generator.add(Dropout(0.2))
        
        generator.add(Conv3D(256, kernel_size=(kernel_sz, kernel_sz, 5), padding='same'))
        generator.add(Activation("relu"))
        generator.add(UpSampling3D(size=(2, 2, 2)))
        generator.add(Conv3D(128, kernel_size=(kernel_sz, kernel_sz, 5), padding='same'))
        generator.add(Activation("relu"))
        generator.add(UpSampling3D(size=(2, 2, 2)))
        generator.add(Conv3D(64, kernel_size=(kernel_sz, kernel_sz, 5), padding='same'))
        generator.add(Activation("relu"))
        #generator.add(UpSampling3D(size=(2, 2, 2)))
        #generator.add(Conv3D(78, kernel_size=(5, 5, 5), padding='same'))
        #generator.add(Activation("relu"))
        generator.add(Conv3D(self.nchan, kernel_size=(kernel_sz, kernel_sz, 5), padding='same', 
            activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.summary()

        return generator

    def buildDiscriminator(self, max_filters, kernel_sz):

        discriminator = Sequential()

        discriminator.add(Conv3D(256, kernel_size=(kernel_sz, kernel_sz, 5), strides=(2,2,2), input_shape=self.image_dimensions, 
            padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Conv3D(128, kernel_size=(kernel_sz,kernel_sz, 5), strides=(2,2,2), padding="same"))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Conv3D(128, kernel_size=(kernel_sz,kernel_sz, 5), strides=(2,2,2), padding="same"))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        #discriminator.add(Conv3D(64, kernel_size=(kernel_sz,kernel_sz, 5), strides=(2,2,2), padding="same"))
        #discriminator.add(LeakyReLU(.2))
        #discriminator.add(Dropout(0.3))

        discriminator.add(Flatten())

        discriminator.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

        discriminator.summary()

        return discriminator

    def trainGAN(self, generator, eps = 0.2, iterations = NUM_ITER, batch_size = BATCH_SIZE, sample_interval = SAMPLE_INT, randomDim = LATENT_VEC_SIZE):
        
        # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
        # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
        # gradient_penalty loss function and is not used.

        positive_y  = np.ones((self.batch_size, 1), dtype=np.float32)
        negative_y  = - positive_y
        dummy_y     = np.zeros((self.batch_size, 1), dtype=np.float32)

        dLosses                     = []
        gLosses                     = []

        for it in range(iterations + 1):
            #print ("Start it %d ----------" % epoch)
            # shuffle Xtrain
            #np.random.shuffle(X_train)
            #print("Epoch: ", epoch)
            #print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))

            nSamples        = batch_size
            

            for j in range(N_CRITIC_ITER):
                noise           = np.random.normal(0, 1, size=[nSamples, randomDim])
                generatedCube   = np.zeros((nSamples, self.nrows,self.ncols, self.nlayers))
                generatedImages = generator.predict(noise)

                firstLayer      = np.round(np.reshape(generatedImages[:,0,:,:], (nSamples, self.nrows*2, self.ncols*2)))
                # first layer
                generatedCube[:,:,:,0] = firstLayer[:,0:2:self.nrows,0:2:self.ncols]

                # Create cube
                for i in range(1,self.nlayers):
                  noise2            = np.random.normal(0, 1, size=[nSamples, randomDim])
                  noise             = noise + eps*noise2
                  generatedImages   = generator.predict(noise)
                  newLayer          = np.reshape(generatedImages[:,0,:,:], (nSamples, self.nrows*2, self.ncols*2))
                  generatedCube[:,:,:,i] = np.round(newLayer[:,0:2:self.nrows,0:2:self.ncols])

                # Insert channel dimension 
                generatedCube                     = generatedCube[:, np.newaxis, :, :, :]

                # ---------------------
                #  1 Train Discriminator
                # ---------------------

                # Select a random batch of images
                #idx         = np.random.randint(0, generatedCube.shape[0], batch_size)
                #image_batch = generatedCube[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim]).astype(np.float32)

                # Train discriminator model
                d_loss = self.discriminator_model.train_on_batch([generatedCube, noise],
                                                             [positive_y, negative_y, dummy_y])

            # ---------------------
            #  2 Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, size=[batch_size, self.latent_dim]).astype(np.float32)
            g_loss = self.generator_model.train_on_batch(noise, positive_y)

            gLosses.append(g_loss)
            dLosses.append(.5*(d_loss[0] + d_loss[1]))
                
            # Print the progress
            print ("Iteration %d, [D loss: %f] [G loss: %f]" % (it, .5*(d_loss[0] + d_loss[1]), g_loss))
                        
            # If at save interval => save generated image samples
            if it % sample_interval == 0:
                #self.saveGenImages(it)
                #self.saveSampleData(it, generatedCube)
                self.saveModels(it)
                #self.plotLoss(it, dLosses, gLosses)

    def saveSampleData(self, epoch, cube, examples=16, dim=(4, 4), figsize=(10, 10)):
        
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

    def saveGenImages(self, epoch, examples=16, dim=(4, 4), figsize=(10, 10)):
        
        noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
        generated_images = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(examples):
            if i < 8:
                plt.subplot(dim[0], dim[1], i+1)
                plt.imshow(generated_images[0, 0, :, :, 2*i], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                plt.title('Layer %d' % (2*i+1) )
            else:
                plt.subplot(dim[0], dim[1], i+1)
                plt.imshow(generated_images[1, 0, :, :, 2*i-16], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                plt.title('Layer %d' % (2*i + 1 - 8) )
        plt.tight_layout()
        plt.savefig('images/wgan_image_epoch_%d.png' % epoch)
        plt.close()


    # Save the generator and discriminator networks (and weights) for later use
    def saveModels(self, epoch):
        self.generator.save('models/wgan3D_gen_ep_%d.h5' % epoch)
        #self.discriminator.save('models/wgan_discriminator_epoch_%d.h5' % epoch)


if __name__ == '__main__':

    # Load dataset
    input_path  = sys.argv[1]
    output_path = sys.argv[2]
    filename    = sys.argv[3]
    max_filters = int(sys.argv[4])
    kernel_size = int(sys.argv[5])
    #num_pix     = int(sys.argv[6])
    eps         = float(sys.argv[6])


    # Check existence of paths and training data
    if not os.path.exists(input_path):
        print('Input path %s does not exist!' %input_path)
        sys.exit(1)
    print('Input path exists: OK')

    if not os.path.exists(output_path):
        print('Output path %s does not exist!' %output_path)
        sys.exit(1)
    print('Output path exists: OK')

    os.chdir(input_path)
    if not os.path.isfile(filename):
        print('Input file %s does not exist!' %filename)
        sys.exit(1)
    print('Input file exists: OK')
    
    # Load dataset
    generator           = kmod.load_model(filename, 
            custom_objects={'wassersteinLoss': wassersteinLoss})

    nx                          = 96
    ny                          = 96
    nz                          = 24
    nchan                       = 1
  
    # Insert channel dimension 
    #X_train                     = X_train[:, np.newaxis, :, :, :]

    # Create class instance
    print('----------------------------')
    print('Set up GAN model')
    print('----------------------------')
    wgan                        = wGAN(nx, ny, nz, nchan, max_filters, kernel_size)

    # Train GAN
    print('----------------------------')   
    print('Train GAN model')
    print('----------------------------')
    # Start training
    os.chdir(output_path)
    wgan.trainGAN(generator, eps, n_epochs = NUM_ITER, batch_size = BATCH_SIZE, sample_interval = SAMPLE_INT, randomDim = LATENT_VEC_SIZE)

