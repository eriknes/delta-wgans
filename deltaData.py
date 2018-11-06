import numpy as np

# Split into train and test data for GAN 
def build_dataset(X, nx, ny, n_test = 0):
  
  m = X.shape[0]
  print("Number of images: " + str(m) )
  
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
  
  X_test  = X_new[m-n_test:m,:,:]
  Y_test  = Y[m-n_test:m]
  
  print("X_train shape: " + str(X_train.shape))
  print("Y_train shape: " + str(Y_train.shape))
  
  return X_train, Y_train, X_test, Y_test