from __future__ import print_function, division

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
LATENT_DIM              = 32
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC_ITER           = 5
NUM_ITER                = 15000
SAMPLE_INT              = 500
ADAM_LR                 = 0.0002
ADAM_B1                 = 0.5
ADAM_B2                 = 0.99

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
    def __init__(self, X_train, n_facies, max_filters, kernel_sz):

        #np.random.seed(1)

        K.set_image_dim_ordering('th')
        self.nrows                  = X_train.shape[2]
        self.ncols                  = X_train.shape[3]
        self.nchan                  = n_facies#X_train.shape[1]
        self.image_dimensions       = (self.nchan, self.nrows, self.ncols)
        
        self.batch_size             = BATCH_SIZE
        self.latent_dim             = LATENT_DIM

        # Adam gradient descent
        #optim               = Adam(lr = 0.0001, beta_1 = 0.5, beta_2 = 0.99)
        optim               = Adam(lr = ADAM_LR, beta_1 = ADAM_B1, beta_2 = ADAM_B2)

        # Build the generator
        self.generator      = self.buildGenerator(max_filters, kernel_sz)
        # Build discriminator
        self.discriminator  = self.buildDiscriminator(max_filters, kernel_sz)
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


        #real_samples                        = Input(shape=X_train.shape[1:])
        real_samples                        = Input(shape=self.image_dimensions)
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

    def buildGenerator(self, max_filters, kernel_sz):

        generator = Sequential()
        generator.add(Dense(max_filters*12*12, input_dim=self.latent_dim, 
            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        #generator.add(LeakyReLU(.2))
        generator.add(Activation("relu"))
        #generator.add(Dropout(0.2))
        generator.add(Reshape((max_filters, 12, 12)))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(int(max_filters / 2), kernel_size=(kernel_sz, kernel_sz), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(int(max_filters / 2), kernel_size=(kernel_sz, kernel_sz), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(int(max_filters / 4), kernel_size=(kernel_sz, kernel_sz), padding='same'))
        generator.add(Activation("relu"))

        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(int(max_filters / 8), kernel_size=(kernel_sz, kernel_sz), padding='same'))
        generator.add(Activation("relu"))
        
        generator.add(Conv2D(self.nchan, kernel_size=(kernel_sz, kernel_sz), padding='same', activation='sigmoid',
            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.summary()

        return generator


    def buildDiscriminator(self, max_filters, kernel_sz):

        # Sequential model
        discriminator = Sequential()

        # First layer is conv2D
        discriminator.add(Convolution2D(max_filters, kernel_size=(kernel_sz, kernel_sz), strides=(2,2), input_shape=self.image_dimensions, 
            padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        # Leaky Relu activation
        discriminator.add(LeakyReLU(.2))
        # Dropout regularization
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(int(max_filters), kernel_size=(kernel_sz, kernel_sz), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(int(max_filters / 2), kernel_size=(kernel_sz, kernel_sz), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(int(max_filters / 4), kernel_size=(kernel_sz, kernel_sz), strides=(2,2), padding="same"))
        #discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Convolution2D(int(max_filters / 8), kernel_size=(kernel_sz, kernel_sz), strides=(2,2), padding="same"))
        #discriminator.add(BatchNormalization(momentum=0.7))
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Flatten())
        
        discriminator.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

        # Print summary of model
        discriminator.summary()

        return discriminator

    def trainGAN(self, X_train, n_facies, iterations = NUM_ITER, batch_size = BATCH_SIZE, sample_interval = SAMPLE_INT):

        positive_y  = np.ones((self.batch_size, 1), dtype=np.float32)
        negative_y  = - positive_y
        dummy_y     = np.zeros((self.batch_size, 1), dtype=np.float32)

        dLosses                     = []
        gLosses                     = []

        for it in range(iterations + 1):
            # shuffle Xtrain
            #np.random.shuffle(X_train)

            #for i in range(batch_count):

            for j in range(N_CRITIC_ITER):

                # ---------------------
                #  1 Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                image_batch = createImageFaciesChannels(X_train[idx], n_facies)
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

            # Loss    
            gLosses.append(g_loss)
            dLosses.append(-d_loss[0] - d_loss[1] - d_loss[2])
                
            # Print the progress
            if it % (sample_interval/10) == 0:
                print ("Iteration %d, [D loss: %f] [G loss: %f]" % (it, -(d_loss[0] + d_loss[1] + d_loss[2]), g_loss))
                    
            # If at save interval => save generated image samples
            if it % sample_interval == 0:
                self.saveGeneratedImages(it, 3)
                self.saveModels(it)
                self.saveLoss(it, dLosses, gLosses)


    def saveGeneratedImages(self, it, examples=3):
        noise = np.random.normal(0, 1, size=[examples, self.latent_dim])
        generated_images = self.generator.predict(noise)
        dim=(examples, self.nchan)

        # Do not display plot
        plt.ioff()
        #plt.figure(figsize=(20,10))
        for i in range(examples):
            for j in range(self.nchan):
                plt.subplot(dim[0], dim[1], i*self.nchan+j+1)
                plt.imshow(generated_images[i, j], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
                plt.title("Facies " + str(j))
        plt.tight_layout()
        plt.savefig('samples/wgan_image_iter_%d.png' % it)
        plt.close()


    # Plot the loss from each batch
    def saveLoss(self, it, dLosses, gLosses):
        discFileName = "samples/disc_loss_it_{0}.csv".format(it)
        genFileName="samples/gen_loss_it_{0}.csv".format(it)
        np.savetxt(discFileName, dLosses, delimiter=",")
        np.savetxt(genFileName, gLosses, delimiter=",")


    def saveModels(self, it):
        self.generator.save('models/generator_it_%d.h5' % it)
        self.discriminator.save('models/discriminator_it_%d.h5' % it)

def createImageFaciesChannels(X, nchan):
    m   = X.shape[0]
    nx  = X.shape[2]
    ny  = X.shape[3]

    X_new       = np.zeros(shape=(m, nchan, nx, ny), dtype='int8')
    for i in range(m):
        Xtemp1 = np.reshape(X[i,0,:,:], (nx,ny))
        for j in range(1,nchan+1):
            Xtemp2                          = np.zeros(Xtemp1.shape)
            Xtemp2[np.where(Xtemp1 == j)]   = 1
            X_new[i,j-1,:,:]                = Xtemp2
    return X_new

# Read csv file
def load_file(fname):
     X = pd.read_csv(fname, header=None, dtype='int8', nrows=15000)
     X = X.values
     #X = X.astype(dtype='int8')
     return X

 # Split into train and test data for GAN 
def build_dataset(input_path, filename, nx, ny):


    os.chdir(input_path)
    X           = load_file(filename)

    m           = X.shape[0]
    print("Number of images in dataset: " + str(m) )

    X           = X.T

    # Random permutation of samples
    p           = np.random.permutation(m)
    X           = X[:,p]
    nchan       = np.size(np.unique(X))
    print("Number of facies in dataset: " + str(nchan-1))

    # Reshape X 
    #X_new       = np.zeros(shape=(m, nchan-1, nx, ny), dtype='int8')
    X_new = np.zeros(shape=(m, 1, nx, ny), dtype='int8')

    for i in range(m):
        X_new[i,0,:,:] = np.reshape(X[:,i],(nx,ny))

    #for i in range(m):
    #    Xtemp1 = np.reshape(X[:,i],(nx,ny)) 
    #    for j in range(1,nchan):
    #        Xtemp2              = np.zeros(Xtemp1.shape)
    #        Xtemp2[np.where(Xtemp1 == j)] = 1
    #        X_new[i,j-1,:,:]      = Xtemp2
    
    # Test generation of batch
    idx = np.random.randint(0, X_new.shape[0], 32)
    image_batch = createImageFaciesChannels(X_new[idx], nchan - 1)
    print("Image batch shape:")
    print(image_batch.shape)

    print("X_train shape: " + str(X_new.shape))

    nchan = nchan - 1

    return X_new, nchan

if __name__ == '__main__':

    # Load dataset
    input_path  = sys.argv[1]
    output_path = sys.argv[2]
    filename    = sys.argv[3]
    max_filters = int(sys.argv[4])
    kernel_size = int(sys.argv[5])
    num_pix     = int(sys.argv[6])

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

    # Build dataset
    print('----------------------------')
    print('Start building dataset')
    print('----------------------------')
    X_train, n_facies     = build_dataset(input_path, filename, num_pix, num_pix)

    # Create class instance
    print('----------------------------')
    print('Set up GAN model')
    print('----------------------------')
    wgan        = wGAN(X_train, n_facies, max_filters, kernel_size)

    # Train GAN
    print('----------------------------')
    print('Train GAN model')
    print('----------------------------')
    os.chdir(output_path)
    wgan.trainGAN(X_train, n_facies)
