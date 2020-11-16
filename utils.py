import os
import itk
import glob
import re
import cv2
import numpy as np
import scipy.stats as sp
import multiprocessing
from skimage import io
from keras import backend as K
from joblib import Parallel, delayed

'''
LOSS FUNCTIONS
'''

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



'''
PREPROCESSING FUNCTIONS
'''

def preprocess_dataset(im_path, img_size=(600,600,700), target_size=(128,128,128),
                       stack=False, save_tiffs=False):
    
    stack_IDs = os.listdir(im_path)
    print('%i volumes found ...' %(len(stack_IDs)))    
    print('3D image stacks to be downsampled to ... (%d, %d, %d)' 
              %(target_size[0], target_size[1], target_size[2]))
    
    num_cores = multiprocessing.cpu_count()
    if num_cores > len(stack_IDs):
        num_cores = len(stack_IDs)
    
    Parallel(n_jobs=num_cores, verbose=50)(delayed(
            process_tiff)(im_path, 
                          img_size=img_size, 
                          target_size=target_size,
                          stack=stack, 
                          save_tiffs=save_tiffs,
                          ID=file,
                          stack_IDs=stack_IDs)for file in range(len(stack_IDs)))
    
def process_tiff(im_path, img_size=(600,600,700), target_size=(128,128,128),
                 stack=False, save_tiffs=False, ID=None, stack_IDs=None):
    
        ID = stack_IDs[ID]
        
        img_in_path = os.path.join(im_path, ID, "Input/")
        imgs_in = load_volume(img_in_path, 
                              size=img_size, 
                              datatype='uint8',
                              stack=stack)
        
        if save_tiffs:
            inarrayout = os.path.join(im_path, ID, ID+'_readin_input.tiff')
            io.imsave(inarrayout, np.transpose(imgs_in.astype('uint8'),(2,0,1)), bigtiff=False)
            

        #  Downsampling image volumes
        imgs_in = resize_stacks(imgs_in.astype('uint8'),  
                                img_size=img_size, 
                                target_size=target_size)
        
        #  Apply local standardisation
        imgs_in = imgs_in.astype('float32')
        for j in range(imgs_in.shape[2]):
            imgs_in[:,:,j] = stdnorm(imgs_in[:,:,j])
        
        #  Threshold data
        lp = sp.scoreatpercentile(imgs_in,5)
        imgs_in[imgs_in < lp] = lp
        up = sp.scoreatpercentile(imgs_in,95)
        imgs_in[imgs_in > up] = up

        #  Normalise data between [0,1]
        imgs_in = norm_data(imgs_in.astype('float32'))  

        inarrayout = os.path.join(im_path, ID, ID+'_input.npy')
        np.save(inarrayout, imgs_in)
        
        if save_tiffs:
            inarrayout = os.path.join(im_path, ID, ID+'_input.tiff')
            io.imsave(inarrayout, np.transpose((imgs_in*255).astype('uint8'),(2,0,1)), bigtiff=False)
            
        print('Saved %s ... ' %(ID))
        

def load_volume(folder, size=(600,600,700), ext='*.tiff', datatype='float32',
                stack=False):
    
    vol = np.empty([size[0], size[1], size[2]], dtype=datatype)
    
    if ext == '*.npy':
        imgs = os.path.join(folder, ext)
        vol = np.load(imgs)
    else:
        if stack:
            # imgs = os.path.join(folder, 'mask'+ext)
            imgs = folder + ext
            vol = (np.array(itk.imread(imgs, itk.F))).astype(datatype)
        else:
            imgs = glob.glob(os.path.join(folder, ext))
            imgs.sort(key=natural_keys)
            for i in range(0,size[2]):
                idx = imgs[i]
                img = cv2.imread(idx, cv2.IMREAD_GRAYSCALE)
                vol[:,:,i] = img
        
    return vol

def resize_stacks(input_imgs, img_size=None, target_size=None):
    
    A = np.empty([target_size[0], target_size[1], img_size[2]],
                           dtype='uint8')
    
    X = np.empty([target_size[0], target_size[1], target_size[2]],
                           dtype='uint8')
    
    #  Assuming input and mask images are equal in size
    if target_size[0] > img_size[0] & target_size[1] > img_size[1]:
        imethod = cv2.INTER_AREA
    else:
        imethod = cv2.INTER_CUBIC

    for j in range(0, img_size[2]):
    
        A[:,:,j] = cv2.resize(input_imgs[:,:,j], (target_size[0], target_size[1]),
                              interpolation = imethod)

    
    if target_size[1] > img_size[1] & target_size[2] > img_size[2]:
        imethod = cv2.INTER_AREA
    else:
        imethod = cv2.INTER_CUBIC

    for j in range(0, target_size[0]):
        
        X[j,] = cv2.resize(A[j,], (target_size[2], target_size[1]),
                           interpolation = imethod)
        
    return X

def resize_volume(img, target_size=None):
    
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='uint8')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='uint8')
    
    for i in range(img.shape[2]):
        arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[0], target_size[1]),
                                 interpolation=cv2.INTER_CUBIC)
        
    for i in range(target_size[0]):
        arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                 interpolation=cv2.INTER_CUBIC)
    
    for i in range(arr2.shape[2]):
        _, arr2[:,:,i] = cv2.threshold(arr2[:,:,i], 127, 255, cv2.THRESH_BINARY)
        
    return arr2

def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def stdnorm(data):
    return (data - np.mean(data)) / np.std(data)

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text