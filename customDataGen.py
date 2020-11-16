import os
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    
    """Generates data for Keras"""
    """This structure guarantees that the network will only train once on each sample per epoch"""

    def __init__(self, list_IDs, im_path, batch_size=4, 
                 img_dim=(128, 128, 128), dim=(256, 256, 256), n_channels=1, 
                 n_classes=2, testing=False):
        
        'Initialization'
        self.img_dim = img_dim
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.im_path = im_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()
        self.testing = testing

        print('Found %d image stacks belonging to %d classes.' %
              (len(self.list_IDs), self.n_classes))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        
        X = load_tensors(self, list_IDs_temp)
                
        return X, X
    
def load_tensors(self, list_IDs_temp):
    
    img_inVol = np.empty([self.batch_size, self.n_channels, self.dim[0], 
                                  self.dim[1], self.dim[2]], dtype='float32')

    for i, ID in enumerate(list_IDs_temp):
        
        img_in_path = os.path.join(self.im_path, ID)
        imgs_in = os.path.join(img_in_path, ID+"_input.npy")
        img_inVol[i, 0,] = np.load(imgs_in)
    
    return img_inVol
