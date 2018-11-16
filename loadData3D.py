import numpy as np

# Read csv file
def load_file(fname, type):
     X = pd.read_csv(fname)
     X = X.values.astype(type)
     #X = X.astype('uint8')
     return X

# Create channels first training dataset
def buildDataset_3D(X, type, nx, ny, nz):

  X                     = load_file(filename)
  
  m                     = X.shape[0]
  print("Number of images: " + str(m) )
  
  X                     = X.T
  
  # Random permutation of samples
  p         = np.random.permutation(m)
  X         = X[:,p]
  
  # Reshape X and crop to 96x96 pixels
  X_train = np.zeros((m,nx,ny,nz))
  for i in range(m):

    Xtemp = np.reshape(X[:,i],(nz,nx,ny))
    X_train[i,:,:,:] = np.moveaxis(Xtemp, 0, -1)
  
  

  print("X_train shape: " + str(X_train.shape))
  
  return X_train